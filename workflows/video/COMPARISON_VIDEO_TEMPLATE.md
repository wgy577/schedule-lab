# 调度对比视频固定模板

后续调度对比视频统一使用 `render_schedule_comparison_template.py`，不再临时裁切旧视频。

## 固定交付参数

- 分辨率：`1920 × 816`
- 帧率：`30 fps`
- 时长：`60 s`
- 总帧数：`1800`
- 编码：`H.264`
- 像素格式：`yuv420p`
- 码率：`2800 kbit/s`
- 布局：左右各 `960 × 816`，使用完全相同的边距、字体、甘特图和甲板尺寸

## 快速生成

默认以 `10 fps` 渲染调度状态，再封装为固定的 `30 fps` 交付视频。对调度可视化足够流畅，同时显著减少 Matplotlib 重绘时间和电脑发热。

```bash
cd "/Users/guangyuwu/Desktop/sortie code/comparision"
python3 schedule_lab/workflows/video/render_schedule_comparison.py \
  --left /path/to/previous.json \
  --right /path/to/optimized.json \
  --output videos/comparison.mp4
```

两侧必须使用同一条共享模拟时钟和相同的甘特横轴范围。默认以较大的 makespan 作为时间轴终点；较短方案完成后保持最终状态，等待较长方案结束，禁止把两个 makespan 分别归一化到完整 60 秒。

模板按“调度文件修改时间 + 模板版本 + 渲染参数 + 共享时间轴终点”缓存单侧视频。后续只修改一个候选方案时，另一侧直接复用，不再重复渲染。

生成前会比较左右调度的规范化 SHA-256、工序集合、机器绑定和时序差异。左右调度相同会直接报错，避免再次生成看似对比、实际相同的视频；输出旁同时保存 `.manifest.json` 审计文件。

如确实需要每个画面均为原生 30 fps，可额外使用 `--render-fps 30`，但生成时间和负载会明显增加。
