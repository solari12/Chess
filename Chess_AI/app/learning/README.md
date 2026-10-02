# Learning trong Chess AI

Tài liệu này mô tả subsystem learning trong `Chess_AI/app/learning/`: Genetic Algorithm Laboratory và pipeline time-management. Chạy mọi lệnh Python từ thư mục gốc `Chess_AI/`, không phải thư mục `app/learning/`.

README backend tổng thể nằm tại [`Chess_AI/README.md`](../../README.md). Tài liệu này tập trung vào code, dữ liệu, model, tests và trạng thái các thí nghiệm learning.

## Tổng quan

Learning có hai nhánh độc lập:

| Nhánh | Mục tiêu | Ảnh hưởng engine mặc định |
|---|---|---|
| `genetic/` | Tiến hóa trọng số quân cờ trong phòng thí nghiệm riêng | Không; dùng evaluator và API riêng |
| `time_management/` | Sinh Teacher target, train/đánh giá dự đoán thời gian search | Không; model hiện vẫn shadow/validation-only |

“Learning” không có nghĩa model tự cập nhật khi chơi. Dataset, nhãn Teacher và artifacts được tạo bởi các lệnh offline có chủ đích. Genetic Lab giữ state trong RAM của backend; time-management lưu JSONL/JSON/model trên đĩa.

## Cấu trúc code

```text
app/learning/
├── genetic/
│   ├── models.py             # chromosome, candidate, config, history/trace
│   ├── population.py         # quần thể ban đầu, giới hạn gene
│   ├── selection.py          # tournament selection
│   ├── crossover.py          # uniform crossover theo từng gene
│   ├── mutation.py           # bounded mutation và mutation events
│   ├── fitness.py            # chơi game, W/D/L, tính fitness
│   └── evolution.py          # điều phối generation và opponent pool
└── time_management/
    ├── generate_dataset.py   # CLI tạo raw dataset Phase 2A
    ├── dataset.py            # feature, telemetry, JSONL và validation
    ├── position_generator.py # sinh vị trí có seed hoặc lấy từ PGN cục bộ
    ├── features.py           # feature từ vị trí hiện tại
    ├── teacher.py            # target/label Phase 2B
    ├── model_features.py     # schema dùng chung khi train và inference
    ├── train_models.py       # median, Gradient Boosting, Random Forest
    ├── evaluate_models.py    # robustness/ablation Phase 2D
    ├── phase2e.py            # low-clock và clock-cap validation
    ├── runtime_shadow.py     # post-search inference/logging, không điều khiển
    ├── evaluate_shadow.py    # tổng hợp runtime shadow log
    ├── run_shadow_sample.py  # chạy search thật để lấy shadow sample
    ├── phase2f1.py           # collection và probe study offline
    ├── phase2f2.py           # held-out Teacher-vs-model validation
    ├── telemetry.py          # kiểu dữ liệu telemetry
    ├── validation.py         # đối chiếu raw/labeled records
    └── models.py             # data structures của dataset
```

`requirements.txt` ở [`Chess_AI/requirements.txt`](../../requirements.txt) khai báo FastAPI, Uvicorn, `python-chess` và scikit-learn; scikit-learn kéo theo NumPy/joblib cần cho model.

Khởi tạo môi trường từ `Chess_AI/` nếu chưa cài dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Time-management pipeline

| Phase | Nội dung | Artifacts chính |
|---|---|---|
| 2A | Lấy mẫu FEN và chạy bounded iterative Alpha-Beta; clock variants synthetic | Raw JSONL và metadata |
| 2B | Heuristic Teacher gán `teacher_time_ms`, depth, confidence | Labeled JSONL và summary |
| 2C | Baseline median, Gradient Boosting, Random Forest; split theo FEN | Model artifacts và evaluation |
| 2D | Nhiều seed, ablation và learning curve | Report, error cases, curve |
| 2E | Kiểm tra clock thấp và safety cap | `time_pressure_*`, `phase2e_clock_*` |
| 2F | Inference sau search và ghi shadow log | `runtime_shadow.jsonl`, report |
| 2F.1 | 108 search thật trên clock/FEN khác nhau và pre-search probe study | `phase2f1_*` |
| 2F.2 | Frozen model so với Teacher trên grouped held-out FEN | `phase2f2_*` |

### Phase 2A–2B: dữ liệu và nhãn

Raw Phase 2A record chứa `sample_id`, FEN/source, clock, position features, tổng search telemetry và depth profile; nó chưa chứa target. Một FEN có thể có nhiều clock variants.

```powershell
python -m app.learning.time_management.generate_dataset `
  --games 10 --sampling-interval 4 --game-depth 3 `
  --game-candidate-count 3 --max-plies 40 `
  --analysis-budget-ms 1000 --max-analysis-depth 6 `
  --remaining-times-ms 300000,120000,30000,10000,3000 `
  --seed 202602 --output data/time_management/learning_raw.jsonl

python -m app.learning.time_management.teacher `
  --input data/time_management/learning_raw.jsonl `
  --output data/time_management/learning_labeled.jsonl `
  --summary data/time_management/learning_teacher_summary.json

python -m app.learning.time_management.validation `
  --raw data/time_management/learning_raw.jsonl `
  --labeled data/time_management/learning_labeled.jsonl `
  --output data/time_management/learning_validation_report.json
```

Teacher là heuristic offline, không phải xác suất thống kê hay thời gian tối ưu đã được chứng minh. Record không đủ search depth/usability có thể mang `label_status=insufficient_data`; model training chỉ lấy record `labeled`.

### Phase 2C–2D: huấn luyện và đánh giá

Target là `teacher_time_ms`. Feature extraction duy nhất dùng chung là `model_features.extract_model_features`; split theo FEN bảo đảm clock variants cùng vị trí không rơi vào train và test cùng lúc.

```powershell
python -m app.learning.time_management.train_models `
  --input data/time_management/learning_labeled.jsonl `
  --output-root data/time_management/learning_experiment `
  --random-state 42 --test-size 0.2

python -m app.learning.time_management.evaluate_models `
  --input data/time_management/learning_labeled.jsonl `
  --output-root data/time_management/learning_experiment/phase2d `
  --seeds 42,43,44,45,46
```

2C lưu median baseline và hai regressors; 2D kiểm tra robustness/ablation. Dùng thư mục mới cho mỗi lần chạy; một số evaluator từ chối ghi đè artifacts cũ. Tạo model mới không tự thay artifact mà runtime shadow đang trỏ tới.

### Model và dataset đang có

Artifact Phase 2C hiện được runtime shadow tham chiếu:

```text
data/time_management/models/phase2c_20261001T122840Z_seed42/
├── random_forest.joblib
├── gradient_boosting.joblib
└── metadata.json
```

Metadata lưu target, thứ tự feature, seed và cấu hình; `model_features.json` ghi feature audit. Không đổi thứ tự feature thủ công: loader xác thực schema và số chiều trước khi inference.

Dataset [`validation_training_dataset.jsonl`](../../data/time_management/validation_training_dataset.jsonl) là labeled Phase 2B đang dùng: 490 records/98 FEN groups; 480 labeled, 10 insufficient; clocks 3,000, 10,000, 30,000, 120,000 và 300,000 ms. Model dùng 480 labeled records: 380 train records/77 FEN groups và 100 test records/20 FEN groups. Một FEN group chỉ có nhãn insufficient nên không thuộc model split.

`teacher_time_ms`, `teacher_depth`, confidence, information gain, label status, depth profile và Teacher summaries là target/diagnostic, không phải feature. Các feature telemetry như completed depth, nodes, elapsed time chỉ có sau bounded search. Vì vậy metric trên telemetry đã có được gọi là **oracle-telemetry evaluation**, không chứng minh pre-search inference đã sẵn sàng.

### Phase 2E–2F.2: clocks, shadow và holdout

Phase 2E kiểm tra clocks từ 30,000 xuống 300 ms và reserve 300 ms. Chạy vào output directory riêng:

```powershell
python -m app.learning.time_management.phase2e `
  --output-dir data/time_management/phase2e_rerun `
  --games 2 --sampling-interval 4 --max-plies 16 `
  --analysis-budget-ms 1000 --max-analysis-depth 6 --seed 42
```

Phase 2F chỉ infer sau khi search kết thúc; prediction không đổi budget/nước đi. 2F.1 thu search runtime thật, nhưng actual duration không được thay cho Teacher target. 2F.2 tái tạo đúng grouped split của model hiện có; không retrain và không chạy Teacher mới.

README gốc có đầy đủ lệnh 2F.1/2F.2 và danh sách artifacts: [Phase 2F.1](../../README.md#phase-2f1--shadow-data-collection), [Phase 2F.2](../../README.md#phase-2f2--held-out-teacher-vs-model-validation).

Kết quả Phase 2F.2: 100 held-out records/20 FEN, overlap 0, leakage audit pass. Oracle RF đạt MAE 88.08 ms, RMSE 125.63 ms, R² 0.512; training-median baseline trên cùng records đạt MAE 144.12 ms, RMSE 207.64 ms. Tuy nhiên probe 25 ms có MAE 189.44 ms trên 15 nhãn khớp, kém matched median baseline 125.40 ms. Tại clock 300 ms reserve để lại 0 usable time nên probe bị bỏ qua. Phase 2F.2 tự nó chưa chứng minh model ưu việt; kết quả controlled Phase 2G được báo ở mục tiếp theo.

### Phase 2G: controlled runtime experiment

Phase 2G chạy baseline 250 ms và model branch (probe 25 ms, frozen Random Forest, safety reserve 300 ms) trên cùng FEN/clock từ `phase2f2_heldout_dataset.jsonl`. Đây là CLI offline; không đổi API hay production search policy. Model budget bị chặn theo clock **sau khi trừ probe time, feature preparation/model inference time và reserve**. Search depth vẫn do iterative deepening quyết định.

```powershell
python -m app.learning.time_management.phase2g run `
  --heldout data/time_management/phase2f2_heldout_dataset.jsonl `
  --output data/time_management/phase2g_runtime_results.jsonl

python -m app.learning.time_management.phase2g report `
  --results data/time_management/phase2g_runtime_results.jsonl `
  --output data/time_management/phase2g_report.json

python -m unittest tests.test_time_management_phase2g -v
```

Report chỉ đưa ra metrics, chênh lệch paired và kiểm tra safety; không tuyên bố branch nào thắng. `GO` chỉ xác nhận thử nghiệm controlled chạy đủ và an toàn, không xác nhận model tốt hơn hay sẵn sàng production.

## Genetic Algorithm Laboratory

Genetic Lab tiến hóa năm integer genes: pawn, knight, bishop, rook, queen. Baseline là `100/320/330/500/900`; king không phải gene và giữ giá trị 20,000. Bounds lần lượt: pawn 50–200, knight/bishop 150–500, rook 300–800, queen 700–1200.

Mỗi candidate đấu BASELINE và opponents không trùng; mỗi opponent được đấu hai màu. Generation đầu lấy opponent ngẫu nhiên trong population; generation tiếp theo lấy từ top năm generation trước. Fitness = wins − losses; draw = 0. Selection là tournament, crossover uniform theo gene, mutation bị chặn trong bounds. Elite chromosomes được giữ và đánh giá lại. Genetic Lab có evaluator riêng, không đổi evaluator game mặc định.

API ở `app/api/genetic_lab.py`, prefix `/api/lab/genetic`:

| Endpoint | Tác dụng |
|---|---|
| `POST /start/stream` | Bắt đầu generation 1, stream SSE |
| `POST /step/stream` | Chạy một generation tiếp theo |
| `GET /state` | Lấy config, population, history và game traces |
| `POST /reset` | Xóa experiment trong process hiện tại |
| `POST /play-move` | Tìm nước với chromosome gửi lên; depth API 1–4 |

Defaults: population 10, 8 games/candidate, 5 seconds of whole-game clock per side, mutation rate 0.15, mutation strength 25, 2 elites, tournament size 3, maximum 200 plies, and 20 generations. The active experiment is held in one backend process; explicit JSON checkpoints and Individual Bank records survive backend restarts. GA does not persist a trained model.

Save Experiment writes a schema-versioned checkpoint under `data/genetic_lab/experiments/<experiment_id>/checkpoint.json`. It stores completed population/history, both Python RNG states, the next individual ID, lineage, and game summaries; move-by-move game traces are omitted. Loading restores the same experiment ID and resumes at the next generation without re-evaluating completed generations. Saving during evaluation checkpoints the last completed generation.

The Individual Bank is stored separately at `data/genetic_lab/individual_bank/individuals.json`; tied-best candidate sets are saved under `data/genetic_lab/candidate_sets/`. Bank records include source experiment/generation, chromosome, fitness and W/D/L, lineage, timestamp, tags, and notes. Starting from selected bank records creates a new experiment, copies their stable IDs and chromosomes as generation-one seeds, evaluates them in the new experiment, and fills any remaining population slots with the new experiment's random seed. The saved bank records remain unchanged.


### GA search and time control

Candidate fitness no longer accepts a fixed search depth. The experiment config exposes `time_control_ms`, a whole-game clock for each side (1,000 to 600,000 ms; the UI displays seconds). Every turn calls the existing Phase 2H `decide_search_budget` with that side's remaining clock. Its bounded probe and frozen model produce a budget subject to the existing 300 ms reserve and fallback behavior. Both the probe and the main search use the active side's candidate/opponent piece weights. The main search uses existing `iterative_search` through its existing maximum-depth limit; the completed depth is recorded per move as observed telemetry.

Each color starts with the same configured clock, and elapsed policy plus search time is charged to that side. Fitness remains wins minus losses, with timeout adjudication and the existing ply-limit draw. The production `TIME_MANAGEMENT_ENABLED` flag still controls only production chess endpoints; GA directly uses the shared Phase 2H policy for its training evaluations. The separate Genetic AI `/play-move` endpoint retains its depth setting because it is for interactive play, not GA fitness evaluation.

## Tests

Chạy từ `Chess_AI/`:

```powershell
# Toàn bộ time-management learning tests
python -m unittest discover -s tests -p "test_time_management*.py" -v

# Test riêng theo phase
python -m unittest tests.test_time_management_dataset -v
python -m unittest tests.test_time_management_teacher -v
python -m unittest tests.test_time_management_models -v
python -m unittest tests.test_time_management_evaluation -v
python -m unittest tests.test_time_management_phase2e -v
python -m unittest tests.test_time_management_shadow -v
python -m unittest tests.test_time_management_phase2f1 -v
python -m unittest tests.test_time_management_phase2f2 -v

# Toàn bộ backend
python -m unittest discover -s tests -v
```

`test_time_management_dataset.py` kiểm tra raw schema/generator; `teacher.py` kiểm tra nhãn và leakage; `models.py` kiểm tra features/training artifact; `evaluation.py` kiểm tra Phase 2D; phase2e/shadow/phase2f1/phase2f2 kiểm tra cap, logging, probe và holdout.

## Quy tắc reproducibility và an toàn dữ liệu

1. Giữ raw Phase 2A bất biến; tạo labeled/evaluation output ở file riêng.
2. Không tách các clock variants của cùng FEN qua hai split.
3. Luôn dùng feature extractor/schema từ model metadata.
4. Không xem runtime `actual_time_ms` là Teacher target.
5. Phase 2F shadow chỉ ghi log; Phase 2H là đường production riêng, mặc định tắt và chỉ bật tường minh.
6. Dùng seed đã ghi trong metadata để tái lập split/vị trí; thời gian thực tế phụ thuộc phần cứng.
7. Trước khi chạy lệnh tạo artifact, chọn output path riêng để giữ lại kết quả cần so sánh.

## Phase 2H: opt-in production integration

`POST /api/ai/move` giữ nguyên response schema. `AIRequest` chấp nhận thêm `remaining_time_ms` tùy chọn; client cũ không cần gửi field này. Policy chỉ áp dụng với `alpha-beta` khi có clock. Minimax và request không có clock tiếp tục dùng fixed-depth path. Chess page hiện gọi `/api/ai/move/timed` cho Alpha-Beta; timed endpoint cũng áp dụng cùng policy khi flag bật và request có `remaining_time_ms`, còn response schema giữ nguyên.

Backend mặc định giữ behavior cũ. Bật thử nghiệm bằng environment variable:

```powershell
$env:TIME_MANAGEMENT_ENABLED = "true"
```

Với clock, runtime chạy probe tối đa 25 ms vì canonical Phase 2C schema có các feature search telemetry cần probe. Model và feature extractor không bị thay đổi. Probe time, feature/model inference time và safety reserve 300 ms đều được trừ trước khi cấp budget chính. Nếu model/policy lỗi, fallback là `min(250 ms, clock còn lại sau policy overhead - 300 ms)`. Khi clock không vượt reserve, không chạy search; API trả một nước hợp lệ ngay để giữ response contract.

`depth` tiếp tục là giới hạn độ sâu cho `iterative_search`; model chỉ dự đoán time budget. Runtime telemetry được ghi qua structured application logs, không thêm trường vào response.

## Phase 2H latency telemetry and cache

Production time-management logs separate `probe_time_ms`, `model_load_time_ms`, `feature_extraction_time_ms`, and `inference_latency_ms`. `inference_latency_ms` measures only the frozen estimator's `predict()` call; feature building, cache lookup/model deserialization, and prediction validation are outside that field. `policy_latency_ms` captures the overall pre-search policy cost and is the quantity used when reserving clock before the main search. The Phase 2C estimator is cached by artifact path for the process lifetime; the first cache miss reports model load time, while later requests report `0` for model loading. No model, feature schema, prediction, or fallback behavior is changed by this instrumentation.

### Phase 2I: paired offline search comparison

After validating the live API path, Phase 2I compares the existing 250 ms baseline with the frozen Phase 2C policy on the same held-out FEN and remaining clock. It does not retrain, modify gameplay, or establish playing strength. The default run selects ten deterministic, evenly spaced held-out records; `--limit 0` uses every held-out row.

```powershell
python -m app.learning.time_management.phase2i run --limit 10
python -m app.learning.time_management.phase2i report
python -m unittest tests.test_time_management_phase2i -v
```

Raw paired metrics are written to `data/time_management/phase2i_paired_results.jsonl`; the aggregate is written to `data/time_management/phase2i_report.json`. The report records search and policy time, predicted/final budget, depth, nodes, material evaluation after each move, paired differences, fallback count, and safety violations. Material evaluation is descriptive only; do not interpret it as a strength result.
