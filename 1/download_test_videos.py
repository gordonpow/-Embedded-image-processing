"""
測試影片下載輔助腳本（使用 yt-dlp）

用法：
    pip install yt-dlp
    python download_test_videos.py --url https://www.youtube.com/watch?v=XXXX
    python download_test_videos.py --urls urls.txt   # 每行一個 URL

下載的影片會自動限制最大 720p 並輸出 mp4 格式至 test_videos/
"""
import argparse
import os
import sys
from pathlib import Path


def download(url: str, out_dir: Path):
    try:
        import yt_dlp
    except ImportError:
        print("請先安裝: pip install yt-dlp")
        sys.exit(1)

    opts = {
        'format': 'best[height<=720][ext=mp4]/best[ext=mp4]/best',
        'outtmpl': str(out_dir / '%(title).50s.%(ext)s'),
        'quiet': False,
        'no_warnings': True,
        'merge_output_format': 'mp4',
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url',  help='Single video URL')
    p.add_argument('--urls', help='Text file with one URL per line')
    p.add_argument('--out',  default='test_videos', help='Output directory')
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = []
    if args.url:
        urls.append(args.url)
    if args.urls and os.path.exists(args.urls):
        urls.extend([line.strip() for line in open(args.urls) if line.strip() and not line.startswith('#')])

    if not urls:
        print("請使用 --url 或 --urls 指定下載來源")
        return

    for u in urls:
        print(f"\n>>> 下載: {u}")
        try:
            download(u, out_dir)
        except Exception as e:
            print(f"  失敗: {e}")


if __name__ == '__main__':
    main()
