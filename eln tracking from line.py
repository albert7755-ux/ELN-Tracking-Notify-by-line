import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import re
from dateutil.relativedelta import relativedelta
from linebot import LineBotApi
from linebot.models import TextSendMessage

# --- 設定網頁 ---
st.set_page_config(page_title="ELN 專業監控戰情室 (LINE版)", layout="wide")

# ==========================================
# 🔐 雲端機密讀取 (只讀取 LINE 設定)
# ==========================================
try:
    LINE_ACCESS_TOKEN = st.secrets["LINE_ACCESS_TOKEN"]
    MY_LINE_USER_ID = st.secrets["MY_LINE_USER_ID"]
except Exception:
    st.error("⚠️ 尚未設定 Secrets！請至 Streamlit Cloud 後台設定 LINE Token 與 UserID。")
    LINE_ACCESS_TOKEN = ""
    MY_LINE_USER_ID = ""
# ==========================================

# --- 側邊欄 ---
with st.sidebar:
    st.header("💬 設定中心")
    if LINE_ACCESS_TOKEN and MY_LINE_USER_ID:
        st.success(f"✅ LINE 系統連線成功")
    else:
        st.error("❌ LINE 設定未完成")

    st.markdown("---")
    st.header("🕰️ 時光機設定")
    simulated_today = st.date_input("設定「今天」日期", datetime.now())
    st.caption(f"模擬日期：{simulated_today.strftime('%Y-%m-%d')}")
    st.info("💡 **精簡版：**\n僅保留 LINE 通知功能，移除所有 Email 相關模組。")

# --- 函數區 ---
def send_line_summary(message_text):
    if not LINE_ACCESS_TOKEN or not MY_LINE_USER_ID:
        st.error("LINE 設定缺失，無法發送")
        return False
    try:
        line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
        line_bot_api.push_message(MY_LINE_USER_ID, TextSendMessage(text=message_text))
        st.toast("✅ LINE 通知已發送！", icon="💬")
        return True
    except Exception as e:
        st.error(f"❌ LINE 發送失敗：{e}")
        return False

def parse_nc_months(ko_type_str):
    if pd.isna(ko_type_str) or str(ko_type_str).strip() == "": return 1 
    match = re.search(r'NC(\d+)', str(ko_type_str), re.IGNORECASE)
    if match: return int(match.group(1))
    return 1 

def clean_percentage(val):
    if pd.isna(val) or str(val).strip() == "": return None
    try:
        s = str(val).replace('%', '').replace(',', '').strip()
        return float(s)
    except: return None

def find_col_index(columns, include_keywords, exclude_keywords=None):
    for idx, col_name in enumerate(columns):
        col_str = str(col_name).strip().lower()
        if exclude_keywords:
            if any(ex in col_str for ex in exclude_keywords): continue
        if any(inc in col_str for inc in include_keywords):
            return idx, col_name
    return None, None

# --- 主畫面 ---
st.title("📊 ELN 結構型商品 - 戰情室 (LINE Only)")

uploaded_file = st.file_uploader("請上傳 Excel (工作表1格式)", type=['xlsx', 'csv'])

if uploaded_file is not None:
    try:
        # 1. 讀取資料
        try:
            df = pd.read_excel(uploaded_file, sheet_name=0, header=0, engine='openpyxl')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)

        if df.iloc[0].astype(str).str.contains("進場價").any():
            df = df.iloc[1:].reset_index(drop=True)
        cols = df.columns.tolist()
        
        # 2. 定位欄位 (移除 Email 搜尋)
        id_idx, _ = find_col_index(cols, ["債券", "代號", "id"])
        if id_idx is None: id_idx = 0
        strike_idx, _ = find_col_index(cols, ["strike", "執行", "履約"])
        ko_idx, _ = find_col_index(cols, ["ko", "提前"], exclude_keywords=["strike", "執行", "ki", "type"])
        ko_type_idx, _ = find_col_index(cols, ["ko類型", "ko type"]) or find_col_index(cols, ["類型", "type"], exclude_keywords=["ki", "ko"])
        ki_idx, _ = find_col_index(cols, ["ki", "下檔"], exclude_keywords=["ko", "type"])
        ki_type_idx, _ = find_col_index(cols, ["ki類型", "ki type"])
        t1_idx, _ = find_col_index(cols, ["標的1", "ticker 1"])
        
        trade_date_idx, _ = find_col_index(cols, ["交易日"])
        issue_date_idx, _ = find_col_index(cols, ["發行日"])
        final_date_idx, _ = find_col_index(cols, ["最終", "評價"])
        maturity_date_idx, _ = find_col_index(cols, ["到期", "maturity"])
        # email_idx 移除了
        name_idx, _ = find_col_index(cols, ["理專", "姓名", "客戶"])

        if t1_idx is None or ko_idx is None:
            st.error("❌ 嚴重錯誤：無法辨識關鍵欄位。")
            st.stop()

        # 3. 建立資料表
        clean_df = pd.DataFrame()
        clean_df['ID'] = df.iloc[:, id_idx]
        clean_df['Name'] = df.iloc[:, name_idx] if name_idx else "客戶"
        
        clean_df['TradeDate'] = pd.to_datetime(df.iloc[:, trade_date_idx], errors='coerce') if trade_date_idx else pd.NaT
        clean_df['IssueDate'] = pd.to_datetime(df.iloc[:, issue_date_idx], errors='coerce') if issue_date_idx else pd.Timestamp.min
        clean_df['ValuationDate'] = pd.to_datetime(df.iloc[:, final_date_idx], errors='coerce') if final_date_idx else pd.Timestamp.max
        clean_df['MaturityDate'] = pd.to_datetime(df.iloc[:, maturity_date_idx], errors='coerce') if maturity_date_idx else pd.NaT
        
        def calc_tenure(row):
            if pd.notna(row['MaturityDate']) and pd.notna(row['IssueDate']):
                days = (row['MaturityDate'] - row['IssueDate']).days
                return f"{int(round(days/30))}個月" 
            return "-"
        clean_df['Tenure'] = clean_df.apply(calc_tenure, axis=1)

        clean_df['KO_Pct'] = df.iloc[:, ko_idx].apply(clean_percentage)
        clean_df['KI_Pct'] = df.iloc[:, ki_idx].apply(clean_percentage)
        clean_df['Strike_Pct'] = df.iloc[:, strike_idx].apply(clean_percentage) if strike_idx else 100.0
        clean_df['KO_Type'] = df.iloc[:, ko_type_idx] if ko_type_idx else ""
        clean_df['KI_Type'] = df.iloc[:, ki_type_idx] if ki_type_idx else "AKI"
        
        for i in range(1, 6):
            if i == 1: tx_idx = t1_idx
            else:
                tx_idx, _ = find_col_index(cols, [f"標的{i}"])
                if tx_idx is None: tx_idx = t1_idx + (i-1)*2
            if tx_idx < len(df.columns):
                clean_df[f'T{i}_Code'] = df.iloc[:, tx_idx]
                if tx_idx + 1 < len(df.columns): clean_df[f'T{i}_Strike'] = df.iloc[:, tx_idx + 1]
                else: clean_df[f'T{i}_Strike'] = 0
            else: clean_df[f'T{i}_Code'] = ""; clean_df[f'T{i}_Strike'] = 0

        clean_df = clean_df.dropna(subset=['ID'])
        
        # 4. 下載股價
        today_ts = pd.Timestamp(simulated_today)
        min_issue_date = clean_df['IssueDate'].min()
        start_date = today_ts - timedelta(days=30) if pd.isna(min_issue_date) else min(min_issue_date, today_ts - timedelta(days=14))
            
        st.info(f"下載美股資料... (回溯至 {start_date.strftime('%Y-%m-%d')}) ☕")
        
        all_tickers = []
        for i in range(1, 6):
            if f'T{i}_Code' in clean_df.columns:
                tickers = clean_df[f'T{i}_Code'].dropna().astype(str).unique().tolist()
                all_tickers.extend(tickers)
        all_tickers = [t.strip() for t in set(all_tickers) if t != 'nan' and str(t).strip() != '']
        
        try:
            history_data = yf.download(all_tickers, start=start_date, end=today_ts + timedelta(days=1))['Close']
        except:
            st.error("美股連線失敗")
            st.stop()

        # 5. 運算邏輯
        results = []
        line_alert_list = []

        for index, row in clean_df.iterrows():
            ko_thresh_val = row['KO_Pct'] if pd.notna(row['KO_Pct']) else 100.0
            ki_thresh_val = row['KI_Pct'] if pd.notna(row['KI_Pct']) else 60.0
            strike_thresh_val = row['Strike_Pct'] if pd.notna(row['Strike_Pct']) else 100.0
            
            ko_thresh = ko_thresh_val / 100.0
            ki_thresh = ki_thresh_val / 100.0
            strike_thresh = strike_thresh_val / 100.0
            nc_months = parse_nc_months(row['KO_Type'])
            nc_end_date = row['IssueDate'] + relativedelta(months=nc_months)
            
            assets = []
            for i in range(1, 6):
                if f'T{i}_Code' not in row: continue
                code = str(row[f'T{i}_Code']).strip()
                try: initial = float(row[f'T{i}_Strike'])
                except: initial = 0
                if code != 'nan' and code != '' and initial > 0:
                    assets.append({'code': code, 'initial': initial, 'strike_price': initial * strike_thresh, 'locked_ko': False, 'hit_ki': False, 'perf': 0.0, 'price': 0.0, 'ko_record': '', 'ki_record': ''})
            
            if not assets: continue

            ticker_data_source = history_data
            
            # 1. 補最新價
            for asset in assets:
                try:
                    if len(all_tickers) == 1: s = ticker_data_source
                    else:
                        if asset['code'] in ticker_data_source.columns: s = ticker_data_source[asset['code']]
                        else: continue
                    valid_s = s[s.index <= today_ts].dropna()
                    if not valid_s.empty:
                        curr = float(valid_s.iloc[-1])
                        asset['price'] = curr
                        asset['perf'] = curr / asset['initial']
                except: asset['price'] = 0

            # 2. 回測
            product_status = "Running"
            early_redemption_date = None
            is_aki = "AKI" in str(row['KI_Type']).upper()
            
            if row['IssueDate'] <= today_ts:
                backtest_data = ticker_data_source[(ticker_data_source.index >= row['IssueDate']) & (ticker_data_source.index <= today_ts)]
                if not backtest_data.empty:
                    for date, prices in backtest_data.iterrows():
                        if product_status == "Early Redemption": break
                        is_post_nc = date >= nc_end_date
                        all_locked = True
                        for asset in assets:
                            try:
                                if len(all_tickers) == 1: price = float(prices)
                                else: price = float(prices[asset['code']])
                            except: price = float('nan')
                            if pd.isna(price) or price == 0:
                                if not asset['locked_ko']: all_locked = False
                                continue
                            perf = price / asset['initial']
                            date_str = date.strftime('%Y/%m/%d')
                            if is_aki and perf < ki_thresh:
                                if not asset['hit_ki']:
                                    asset['hit_ki'] = True
                                    asset['ki_record'] = f"@{price:.2f} ({date_str})"
                            if not asset['locked_ko']:
                                if is_post_nc and perf >= ko_thresh:
                                    asset['locked_ko'] = True 
                                    asset['ko_record'] = f"@{price:.2f} ({date_str})"
                            if not asset['locked_ko']: all_locked = False
                        if all_locked:
                            product_status = "Early Redemption"
                            early_redemption_date = date

            # 3. 整理結果
            locked_list = []; waiting_list = []; hit_ki_list = []; shadow_ko_list = []
            detail_cols = {}

            for i, asset in enumerate(assets):
                if asset['price'] > 0:
                    if not is_aki and asset['perf'] < ki_thresh: 
                        asset['hit_ki'] = True
                        asset['ki_record'] = f"@{asset['price']:.2f} (EKI)"
                    if asset['perf'] >= ko_thresh and not asset['locked_ko']:
                        shadow_ko_list.append(asset['code'])

                if asset['locked_ko']: locked_list.append(asset['code'])
                else: waiting_list.append(asset['code'])
                if asset['hit_ki']: hit_ki_list.append(asset['code'])
                
                p_pct = round(asset['perf']*100, 2) if asset['price'] > 0 else 0.0
                status_icon = "✅" if asset['locked_ko'] else "⚠️" if asset['hit_ki'] else ""
                price_display = round(asset['price'], 2) if asset['price'] > 0 else "N/A"
                
                cell_text = f"【{asset['code']}】\n原: {asset['initial']}\n現: {price_display}\n({p_pct}%) {status_icon}"
                if asset['locked_ko']: cell_text += f"\nKO {asset['ko_record']}"
                if asset['hit_ki']: cell_text += f"\nKI {asset['ki_record']}"
                detail_cols[f"T{i+1}_Detail"] = cell_text

            hit_any_ki = any(a['hit_ki'] for a in assets)
            all_above_strike_now = all((a['perf'] >= strike_thresh if a['price'] > 0 else False) for a in assets)
            
            valid_assets = [a for a in assets if a['perf'] > 0]
            if valid_assets:
                worst_asset = min(valid_assets, key=lambda x: x['perf'])
                worst_perf = worst_asset['perf']
                worst_code = worst_asset['code']
                worst_strike_price = worst_asset['strike_price']
            else:
                worst_perf = 0; worst_code = "N/A"; worst_strike_price = 0
            
            final_status = ""
            line_status_short = ""

            if today_ts < row['IssueDate']:
                final_status = "⏳ 未發行"
            elif product_status == "Early Redemption":
                final_status = f"🎉 提前出場\n({early_redemption_date.strftime('%Y-%m-%d')})"
                line_status_short = "🎉 已KO提前出場"
            elif pd.notna(row['ValuationDate']) and today_ts >= row['ValuationDate']:
                if all_above_strike_now:
                     final_status = "💰 到期獲利\n(全數 > 執行價)"
                     line_status_short = "💰 到期獲利"
                elif hit_any_ki:
                     final_status = f"😭 到期接股\n{worst_code} @ {round(worst_strike_price, 2)}"
                     line_status_short = f"😭 到期接股 ({worst_code})"
                else:
                     final_status = "🛡️ 到期保本\n(未破KI)"
                     line_status_short = "🛡️ 到期保本"
            else:
                if today_ts < nc_end_date:
                    final_status = f"🔒 NC閉鎖期\n(至 {nc_end_date.strftime('%Y-%m-%d')})"
                    if shadow_ko_list: final_status += f"\n(目前 {len(shadow_ko_list)} 支 > KO價)"
                else:
                    if not waiting_list: final_status = "👀 比價中"
                    else:
                        wait_str = ",".join(waiting_list)
                        final_status = f"👀 比價中\n⏳等待: {wait_str}"
                        if locked_list: final_status += f"\n✅已鎖: {','.join(locked_list)}"
                if hit_any_ki:
                    final_status += f"\n⚠️ KI已破: {','.join(hit_ki_list)}"
                    line_status_short = f"⚠️ KI 已破 ({','.join(hit_ki_list)})"

            if line_status_short:
                line_alert_list.append(f"● {row['ID']} ({row['Name']}): {line_status_short}")

            trade_date_str = row['TradeDate'].strftime('%Y-%m-%d') if pd.notna(row['TradeDate']) else "-"
            issue_date_str = row['IssueDate'].strftime('%Y-%m-%d') if pd.notna(row['IssueDate']) else "-"
            val_date_str = row['ValuationDate'].strftime('%Y-%m-%d') if pd.notna(row['ValuationDate']) else "-"
            mat_date_str = row['MaturityDate'].strftime('%Y-%m-%d') if pd.notna(row['MaturityDate']) else "-"

            row_res = {
                "債券代號": row['ID'], "天期": row['Tenure'], "收件人": row['Name'],
                "狀態": final_status, "最差表現": f"{round(worst_perf*100, 2)}%",
                "KO設定": f"{ko_thresh_val}%", "KI設定": f"{ki_thresh_val}%", "執行價": f"{strike_thresh_val}%",
                "交易日": trade_date_str, "發行日": issue_date_str, "最終評價": val_date_str, "到期日": mat_date_str
            }
            row_res.update(detail_cols)
            results.append(row_res)

        if not results:
            st.warning("⚠️ 無資料")
        else:
            final_df = pd.DataFrame(results)
            st.subheader("📋 專業監控列表")
            
            def color_status(val):
                if "提前" in str(val) or "獲利" in str(val): return 'background-color: #d4edda; color: green'
                if "接股" in str(val) or "KI" in str(val): return 'background-color: #f8d7da; color: red'
                if "未發行" in str(val) or "NC" in str(val): return 'background-color: #fff3cd; color: #856404'
                return ''

            t_cols = [c for c in final_df.columns if '_Detail' in c]; t_cols.sort()
            display_cols = ['債券代號', '天期', '狀態', '最差表現'] + t_cols + ['KO設定', 'KI設定', '執行價', '交易日', '發行日', '最終評價', '到期日']
            column_config = {
                "狀態": st.column_config.TextColumn("目前狀態摘要", width="large"),
                "債券代號": st.column_config.TextColumn("代號", width="small"),
                "天期": st.column_config.TextColumn("天期", width="small"),
                "KO設定": st.column_config.TextColumn("KO", width="small"),
                "KI設定": st.column_config.TextColumn("KI", width="small"),
                "執行價": st.column_config.TextColumn("Strike", width="small"),
                "最差表現": st.column_config.TextColumn("Worst Of", width="small"),
            }
            for i, c in enumerate(t_cols): column_config[c] = st.column_config.TextColumn(f"標的 {i+1} (原始/現價/狀態)", width="large")

            st.dataframe(final_df[display_cols].style.applymap(color_status, subset=['狀態']), use_container_width=True, column_config=column_config, height=600, hide_index=True)
            
            # --- LINE 發送按鈕 (單一按鈕) ---
            st.markdown("### 📢 通知操作")
            if st.button("📲 發送 LINE 摘要給自己 (今日重點)", type="primary"):
                if line_alert_list:
                    summary_text = f"【ELN 戰情快報】\n📅 模擬日期: {simulated_today.strftime('%Y/%m/%d')}\n----------------\n" + "\n".join(line_alert_list)
                    send_line_summary(summary_text)
                else:
                    send_line_summary(f"【ELN 戰情快報】\n📅 {simulated_today.strftime('%Y/%m/%d')}\n----------------\n今日無特殊事件 (KO/KI/到期)，一切安好。")

    except Exception as e:
        st.error(f"發生錯誤：{e}")
else:
    st.info("👆 請上傳 Excel")
