"""
AI 文本生成系統 - 使用三種不同的 Prompt 模板
支援 OpenAI API 或本地 LLM
"""

import os
import csv
import json
import random
from typing import List, Dict, Optional
from openai import OpenAI
import pandas as pd

# 三種 Prompt 模板
PROMPT_TEMPLATES = {
    "A_instruction": """請以正式、學術的語氣，撰寫一篇關於「{topic}」的文章。

要求：
1. 第一段：介紹主題的背景與重要性
2. 第二段：深入分析主要觀點與論證
3. 第三段：總結並提出未來展望

請確保文章結構清晰、邏輯嚴謹、用詞精準。""",

    "B_narrative": """寫一篇關於「{topic}」的文章，用大學生寫作業的風格，口語化一點，不要太正式。

可以分享你的想法、經驗，或者你對這個主題的看法。就像在跟朋友聊天一樣，但還是要有點內容。""",

    "C_role": """你是一位充滿情感的詩人/作家，請以「{topic}」為主題，寫一篇富有情感與個人色彩的文章。

用你的心靈去感受這個主題，用文字表達你的情緒、想像與創意。不要拘泥於格式，讓文字自然流動。"""
}


class AITextGenerator:
    """AI 文本生成器"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        base_url: Optional[str] = None
    ):
        """
        Args:
            api_key: OpenAI API key（如果使用 OpenAI）
            model: 模型名稱
            base_url: 自訂 API base URL（用於本地 LLM）
        """
        self.model = model
        if api_key or base_url:
            self.client = OpenAI(
                api_key=api_key or "dummy",
                base_url=base_url
            )
        else:
            self.client = None
            print("⚠️  未設定 API，將使用模擬生成模式")
    
    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """生成單篇文本"""
        if not self.client:
            # 模擬生成（用於測試）
            return f"[模擬生成文本 - Prompt: {prompt[:50]}...]"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位專業的寫作助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ 生成失敗: {e}")
            return f"[生成錯誤: {str(e)}]"
    
    def generate_batch(
        self,
        topics: List[str],
        prompt_type: str,
        num_per_topic: int = 10,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> List[Dict]:
        """批量生成文本"""
        results = []
        template = PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES["A_instruction"])
        
        for topic in topics:
            print(f"📝 生成 {prompt_type} - 主題: {topic}")
            for i in range(num_per_topic):
                prompt = template.format(topic=topic)
                text = self.generate_text(prompt, temperature, max_tokens)
                
                results.append({
                    "text": text,
                    "topic": topic,
                    "prompt_type": prompt_type,
                    "label": "AI"  # AI 生成的標籤
                })
                
                if (i + 1) % 5 == 0:
                    print(f"   ✓ 已完成 {i + 1}/{num_per_topic}")
        
        return results


def load_topics(file_path: str = "data/topics.txt") -> List[str]:
    """載入主題列表"""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            topics = [line.strip() for line in f if line.strip()]
        return topics
    else:
        # 預設主題
        default_topics = [
            "人工智慧的未來發展",
            "環境保護與永續發展",
            "遠距工作的優缺點",
            "教育改革的必要性",
            "科技對生活的影響",
            "健康飲食的重要性",
            "閱讀習慣的培養",
            "旅遊的意義與價值",
            "音樂對情緒的影響",
            "運動與身心健康"
        ]
        # 建立 topics.txt
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(default_topics))
        return default_topics


def generate_ai_texts(
    output_path: str = "data/generated_ai.csv",
    num_per_prompt: int = 15,
    temperature: float = 0.7,
    max_tokens: int = 500,
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    base_url: Optional[str] = None
):
    """
    主函數：生成 AI 文本並儲存為 CSV
    
    Args:
        output_path: 輸出 CSV 路徑
        num_per_prompt: 每個 prompt 類型生成的文章數
        temperature: 生成溫度
        max_tokens: 最大 token 數
        api_key: OpenAI API key
        model: 模型名稱
        base_url: 自訂 API base URL
    """
    print("🚀 開始生成 AI 文本...")
    
    # 載入主題
    topics = load_topics()
    print(f"📚 載入 {len(topics)} 個主題")
    
    # 初始化生成器
    generator = AITextGenerator(api_key=api_key, model=model, base_url=base_url)
    
    # 生成三種 prompt 類型的文本
    all_results = []
    for prompt_type in ["A_instruction", "B_narrative", "C_role"]:
        num_per_topic = max(1, num_per_prompt // len(topics))
        results = generator.generate_batch(
            topics=topics,
            prompt_type=prompt_type,
            num_per_topic=num_per_topic,
            temperature=temperature,
            max_tokens=max_tokens
        )
        all_results.extend(results)
        print(f"✅ {prompt_type}: 生成 {len(results)} 篇")
    
    # 儲存為 CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"💾 已儲存至 {output_path}")
    print(f"📊 總計生成 {len(all_results)} 篇 AI 文本")
    
    return df


if __name__ == "__main__":
    # 從環境變數讀取 API key（如果有的話）
    api_key = os.getenv("OPENAI_API_KEY")
    
    # 生成文本（可調整參數）
    generate_ai_texts(
        output_path="data/generated_ai.csv",
        num_per_prompt=15,  # 每個 prompt 類型生成 15 篇
        temperature=0.7,
        max_tokens=500,
        api_key=api_key,
        model="gpt-3.5-turbo"
    )

