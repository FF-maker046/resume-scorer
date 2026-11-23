"""
Local development server launcher for Resume Scorer.
Run with: py run.py  or  python run.py
"""
import uvicorn

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🚀 Resume Scorer 简历评分系统正在启动...")
    print("  🌐 本地访问地址: http://127.0.0.1:8000")
    print("  💡 按 Ctrl + C 可停止服务")
    print("=" * 55 + "\n")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
