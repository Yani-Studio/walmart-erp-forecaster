import pandas as pd
import json

def extract_data():
    csv_path = 'results/current_best_ensemble_v4.csv'
    try:
        df = pd.read_csv(csv_path)
        # Take a random sample of 200 items to show in the dashboard
        sample_df = df.sample(n=200, random_state=42)
        
        data_list = []
        for _, row in sample_df.iterrows():
            data_list.append({
                'item': row['id'],
                'forecast': round(float(row['simple_avg_pred']), 2)
            })
            
        # Write to a JS file so the browser can load it without a web server (avoid CORS)
        js_content = f"const realData = {json.dumps(data_list, indent=2)};"
        
        with open('v2/dashboard/real_data.js', 'w') as f:
            f.write(js_content)
        print("Successfully extracted 200 real predictions into v2/dashboard/real_data.js")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    extract_data()
