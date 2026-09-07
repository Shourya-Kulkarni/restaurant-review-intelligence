import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

def generate_business_brief(evaluated_csv_path, output_report_path):
    load_dotenv()
    
    if "GROQ_API_KEY" not in os.environ:
        print("Error: GROQ_API_KEY environment variable not found in .env file.")
        return

    df = pd.read_csv(evaluated_csv_path)
    client = Groq()
    
    cluster_names = {
        0: "Front Counter & Staff Interactions",
        1: "Kitchen Execution & Food Consistency",
        2: "Pricing, Upcharges, & Operational Logistics"
    }
    
    full_report = "# 📋 Strategic Operations Brief: Spice Up Thai\n"
    full_report += "Generated via NLP Cluster Analysis (UMAP + K-Means + Groq API)\n\n"
    full_report += "--- \n\n"
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        reviews_text = ""
        
        for idx, row in cluster_df.iterrows():
            reviews_text += f"- [Rating: {row['rating']}★] {row['text']}\n"
            
        prompt = f"""
        You are an expert restaurant operations consultant specializing in micro-businesses.
        You are reviewing a specific cluster of negative customer reviews for a local Thai restaurant.
        
        CRITICAL CONTEXT: This restaurant is run entirely by a tight team of ONLY TWO PEOPLE. They handle front-of-house, cooking, phone calls, online orders, and cleaning simultaneously. They are experiencing severe burnout.
        
        CLUSTER THEME: {cluster_names[cluster_id]}
        TOTAL REVIEWS IN THIS CLUSTER: {len(cluster_df)}
        
        RAW CUSTOMER REVIEWS FOR THIS CLUSTER:
        {reviews_text}
        
        Based ONLY on the reviews provided above, write a brief, highly actionable strategic memo for the owners.
        Your tone MUST be incredibly empathetic, supportive, and constructive. Do not lecture them. 
        Acknowledge their extreme multi-tasking constraints and frame the solutions around mitigating burnout and streamlining operations.
        
        Format your response strictly using these Markdown headers:
        ### 🔍 The Core Bottleneck
        (Summarize what pattern is happening across these specific reviews in 2 sentences)
        
        ### 💡 The 2-Person Reality
        (Explain sympathetically why this issue is happening as a natural side effect of being understaffed or burnt out)
        
        ### 🛠️ Low-Friction Solutions
        (Provide 2-3 specific, low-effort adjustments they can make immediately to prevent this issue without adding to their workload)
        """
        
        print(f"Sending Cluster {cluster_id} data to Groq API...")
        
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0.2,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        full_report += f"## 🔴 Cluster {cluster_id}: {cluster_names[cluster_id]}\n"
        full_report += f"**Impact Snapshot:** {len(cluster_df)} modern negative reviews grouped here.\n\n"
        full_report += response.choices[0].message.content + "\n\n"
        full_report += "---\n\n"

    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(full_report)
        
    print(f"\n🎉 Success! Executive summary report saved to {output_report_path}")

if __name__ == "__main__":
    generate_business_brief(
        evaluated_csv_path="data/processed/clustered_reviews.csv",
        output_report_path="data/processed/business_strategic_brief.md"
    )
