"""대시보드를 내 컴퓨터에서 띄우는 작은 서버 (선택 사항).

dashboard/index.html 은 그냥 더블클릭해도 열린다. 이 서버는 'AI로 문장 다듬기' 버튼을 쓰고 싶을 때만 필요하다.
  1) pip install flask
  2) 환경변수 ANTHROPIC_API_KEY 에 Anthropic API 키를 넣는다 (없으면 버튼이 숨겨지고 규칙 기반 브리핑만 나온다)
  3) python app.py  →  http://localhost:5000

주의: /api/briefing 은 실제 API 키로 시험해 보지 못했다. 처음 쓸 때 한 번 확인할 것.
"""
import json
import os
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, request, send_file

ROOT = Path(__file__).parent
MODEL = os.environ.get("BRIEFING_MODEL", "claude-haiku-4-5")
app = Flask(__name__)


@app.get("/")
def index():
    return send_file(ROOT / "dashboard" / "index.html")


@app.get("/api/briefing")
def briefing_available():
    return jsonify(available=bool(os.environ.get("ANTHROPIC_API_KEY")))


@app.post("/api/briefing")
def briefing():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return jsonify(error="ANTHROPIC_API_KEY 가 설정되지 않았습니다."), 503
    prompt = (request.get_json(silent=True) or {}).get("prompt", "")
    if not prompt or len(prompt) > 8000:
        return jsonify(error="prompt 가 없거나 너무 깁니다."), 400
    body = json.dumps({"model": MODEL, "max_tokens": 700, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.load(r)
        return jsonify(text="".join(b.get("text", "") for b in out.get("content", [])))
    except Exception as e:   # 키 오류·네트워크 오류 등
        return jsonify(error=str(e)), 502


if __name__ == "__main__":
    app.run(port=5000, debug=False)
