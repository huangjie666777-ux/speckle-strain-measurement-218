# speckle_strain218

散斑图（DIC）面内位移 / 应变测量后端。输入同尺寸 8 位灰度 PNG 参考图与变形图，逐点做子区数字图像相关，输出位移场与 Green-Lagrange 应变场。纯 HTTP 后端，无前端。

## 运行

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn speckle_strain218.app:app --host 127.0.0.1 --port 8000
```

## API

### `POST /measure`

multipart/form-data：

| 字段 | 类型 | 约束 |
|---|---|---|
| `reference` | PNG 文件 | 8 位灰度，各边 ≤ 512 px |
| `deformed` | PNG 文件 | 与参考图同尺寸、同要求 |
| `roi_x, roi_y, roi_w, roi_h` | int | 矩形 ROI，须在图内且不小于子区 |
| `mm_per_pixel` | float | 正数，毫米/像素 |
| `subset` | int | 奇数子区边长，≥ 3 |
| `step_x, step_y` | int | 网格步距，≥ 1 |
| `search_radius` | int | 整数搜索半径，0–16 |
| `max_iter` | int | 最大迭代数，1–100 |

网格点数（ROI 内按子区半宽收缩后按步距布点）不得超过 400。非法图像或参数返回 `422` 并附原因，整请求拒绝；请求间无共享状态。

成功返回 `application/zip`：

- `points.csv` — 逐点结果，保留全部请求网格点：`point_id,row,col,x_px,y_px,x_mm,y_mm,valid,u_px,v_px,u_mm,v_mm,zncc,iterations,converged,strain_valid,exx,eyy,exy,failure_reason`
- `valid_mask.png` — 8 位灰度掩膜（行×列 = 网格形状，255=有效）
- `params.json` — 请求参数、坐标/单位说明、有效点统计与失败原因汇总

### `GET /health`

返回 `{"status": "ok"}`。

## 坐标与单位

- 坐标系：x 向右、y 向下，原点为左上角像素中心；仅测面内变形。
- 位移：CSV 同时给出像素（`u_px,v_px`）与毫米（`u_mm,v_mm`，由 `mm_per_pixel` 换算）。
- 应变：无量纲。由有效位移的 3×3 邻域二维线性最小二乘拟合位移梯度 H，构造变形梯度 F = I + H，输出 Green-Lagrange 应变 E = (FᵀF − I)/2 的 `exx`、`eyy`、`exy`；`exy` 为张量剪切量 E₁₂（工程剪应变为 2·exy）。邻域内有效点少于 3 个或共线时应变无效。

## 算法

1. 每个网格点独立测量（非整图配准）：先在 ±search_radius 内做整数平移 ZNCC 搜索取初值。
2. 以六参数局部仿射 warp（平移 + 一阶形函数）做前加式 Gauss-Newton 亚像素优化，最小化零均值归一化残差（ZNSSD），双线性插值采样；零均值归一化使其对亮度偏移与正比例增益不变。
3. 逐点失效（不补零位移）：`flat_texture`（参考子区灰度标准差过低）、`subset_out_of_bounds` / `search_out_of_bounds` / `warp_out_of_bounds`（越界）、`degenerate_intensity` / `degenerate_hessian`（退化）、`not_converged`（超过 max_iter）、`low_correlation`（ZNCC < 0.8）。失效点位移/应变留空，不插补。

## 适用范围

- 面内小/中等变形：子区内变形可近似一阶仿射；应变在 3×3 邻域内近似线性。
- 散斑需随机、对比度适中；平坦或饱和区域会逐点失效。
- 刚体位移幅值应小于 search_radius（整数搜索范围）。

## 可复现合成示例

```bash
.venv/bin/python examples/make_synthetic.py examples/out
# 生成 reference.png / deformed.png：已知平移 (2,1) px +
# 位移梯度 [[0.01, 0.002],[0, -0.005]]，并加 1.05 增益与 +8 亮度偏移

curl -s -o dic_result.zip -w '%{http_code}\n' \
  -F reference=@examples/out/reference.png \
  -F deformed=@examples/out/deformed.png \
  -F roi_x=20 -F roi_y=20 -F roi_w=160 -F roi_h=160 \
  -F mm_per_pixel=0.01 -F subset=21 -F step_x=20 -F step_y=20 \
  -F search_radius=8 -F max_iter=50 \
  http://127.0.0.1:8000/measure
.venv/bin/python -c "import zipfile; print(zipfile.ZipFile('dic_result.zip').namelist())"
```

## 测试

```bash
.venv/bin/python -m pytest tests -q
```
