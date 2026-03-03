from huggingface_hub import snapshot_download

print("开始下载 MOMENT-1-small...")
try:
    model_path = snapshot_download(
        repo_id="AutonLab/MOMENT-1-small",
        local_dir="./MOMENT-1-small",
        local_dir_use_symlinks=False,  # 下载真实文件，而不是链接
        resume_download=True           # 支持断点续传
    )
    print(f"下载成功！模型已保存在: {model_path}")
except Exception as e:
    print(f"下载出错: {e}")