video_preprocess/
├── 01_video_segmentation.py         # 视频分割并输出标注视频
├── 02_extract_frames.py             # 逐帧拆分保存（原图、可视化、掩膜、CSV）
├── 03_plot_height_diff.py           # 绘制气液‑液液高度差彩色曲线（带分离终点）
├── 04_plot_solid_timeline.py        # 绘制固‑气/液‑固出现时间轴（甘特图）
└── 05_plot_height_diff_corrected.py # 绘制高度差修正曲线（带drop修正）