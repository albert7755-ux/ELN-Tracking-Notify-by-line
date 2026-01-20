import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import re
from dateutil.relativedelta import relativedelta
from linebot import LineBotApi
from linebot.models import TextSendMessage

# --- 設定網頁 ---
st.set_page_config(page_title="ELN 智能戰情室 (KI優先版)", layout="wide")

# ==========================================
# 🔐 雲端機密讀取 (LINE)
# ==========================================
try:
    LINE_ACCESS_TOKEN = st.secrets.get("LINE_ACCESS_TOKEN", "")
    MY_LINE_USER_ID = st.secrets.get("MY_LINE_USER_ID", "")
except Exception:
    st.error("⚠️ Secrets 設定不完整！")
    LINE_ACCESS_TOKEN = ""
    MY_LINE_USER_ID = ""

# ==========================================
# 🔄 狀態初始化
# ==========================================
if 'last_processed_file' not in st.session_state:
    st.session_state['last_processed_file'] = None
if 'is_sent' not in st.session_state:
    st.session_state['is_sent'] = False

# --- 側邊欄 ---
with st.sidebar:
    st.header("💬 設定中心")
    if LINE_ACCESS_TOKEN:
        st.success(f"✅ LINE 連線 OK")
    else:
        st.error("❌ LINE 未設定")

    st.markdown("---")
    real_today = datetime.now()
    st.info(f"📅 今天日期：{real_today.strftime('%Y-%m-%d')}")
    st.caption("鎖定為真實日期")
    
    st.markdown("---")
    st.header("🔔 通知過濾")
    lookback_days = st.slider("只通知幾天內發生的事件？", min_value=1, max_value=30, value=3)
    notify_ki_daily = st.checkbox("KI/DRA 是否每天提醒？", value=True, help="打勾：持續跌破期間每天都會通知。")

    st.warning("⚠️ **安全模式**\nKI 跌破訊息將強制置頂顯示，不會被 DRA 狀態掩蓋。")

# --- 函數區 ---

def clean_ticker_symbol(ticker):
    if pd.isna(ticker): return ""
    t = str(ticker).strip().upper()
    t = re.sub(r'\s+(UW|UN|UQ|UP|US)$', '', t)
    if t.endswith(" JT"): return t.replace(" JT", ".T") 
    if t.endswith(" TT"): return t.replace(" TT", ".TW") 
    if t.endswith(" HK"): return t.replace(" HK", ".HK") 
    return t

def send_line_push(target_user_id, message_text):
    if not LINE_ACCESS_TOKEN or not target_user_id: return False
    try:
        uid = str(target_user_id).strip()
        if len(uid) < 10 or not (uid.startswith("U") or uid.startswith("C")): 
            return False
        line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
        line_bot_api.push_message(uid, TextSendMessage(text=message_text))
        return True
    except Exception as e:
        print(f"LINE 發送失敗 ({target_user_id}): {e}")
        return False

def parse_nc_months(ko_type_val):
    s = str(ko_type_val).upper().strip()
    if pd.isna(ko_type_val) or s == "" or s == "NAN": return 1 
    match = re.search(r'(?:NC|LOCK|NON-CALL)\s*[:\-]?\s*(\d+)', s)
    if match: return int(match.group(1))
    if "DAILY" in s: return 1
    return 1

def calculate_maturity(row, issue_date_col, tenure_col):
    if 'MaturityDate' in row and pd.notna(row['MaturityDate']):
        return row['MaturityDate']
    issue_date = row.get(issue_date_col)
    tenure_str = str(row.get(tenure_col, ""))
    if pd.isna(issue_date) or issue_date == pd.NaT: return pd.NaT
    try:
        months_to_add = 0
        match_m = re.search(r'(\d+)\s*M', tenure_str, re.IGNORECASE)
        match_y = re.search(r'(\d+)\s*Y', tenure_str, re.IGNORECASE)
        if match_m: months_to_add = int(match_m.group(1))
        elif match_y: months_to_add = int(match_y.group(1)) * 12
        elif tenure_str.isdigit(): months_to_add = int(tenure_str)
        if months_to_add > 0: return issue_date + relativedelta(months=months_to_add)
    except: pass
    return pd.NaT

def clean_percentage(val):
    if pd.isna(val) or str(val).strip() == "": return None
    try:
        s = str(val).replace('%', '').replace(',', '').strip()
        return float(s)
    except: return None

def clean_name_str(val):
    if pd.isna(val): return "貴賓"
    s = str(val).strip()
    if s.lower() == 'nan' or s == "": return "貴賓"
    return s

def find_col_index(columns, include_keywords, exclude_keywords=None):
    for idx, col_name in enumerate(columns):
        col_str = str(col_name).strip().lower().replace(" ", "")
        if exclude_keywords:
            if any(ex in col_str for ex in exclude_keywords): continue
        if any(inc in col_str for inc in include_keywords):
            return idx, col_name
    return None, None

# --- 主畫面 ---
st.title("📊 ELN 智能戰情室 - KI修復版")

uploaded_file = st.file_uploader("請上傳 Excel", type=['xlsx', 'csv'], key="uploader")

if uploaded_file:
    if st.session_state['last_processed_file'] != uploaded_file.name:
        st.session_state['last_processed_file'] = uploaded_file.name
        st.session_state['is_sent'] = False

if uploaded_file is not None:
    try:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=0, header=0, engine='openpyxl')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)

        df = df.dropna(how='all')
        if df.iloc[0].astype(str).str.contains("進場價").any():
            df = df.iloc[1:].reset_index(drop=True)
        cols = df.columns.tolist()
        
        # 欄位定位
        id_idx, _ = find_col_index(cols, ["債券", "代號", "id", "商品代號"]) or (0, "")
        type_idx, _ = find_col_index(cols, ["商品類型", "ProductType", "type"], exclude_keywords=["ko", "ki"]) 
        strike_idx, _ = find_col_index(cols, ["strike", "執行", "履約"])
        ko_idx, _ = find_col_index(cols, ["ko", "提前"], exclude_keywords=["strike", "執行", "ki", "type"])
        ko_type_idx, _ = find_col_index(cols, ["ko類型", "kotype"]) or find_col_index(cols, ["類型", "type"], exclude_keywords=["ki", "ko", "商品"])
        ki_idx, _ = find_col_index(cols, ["ki", "下檔"], exclude_keywords=["ko", "type"])
        ki_type_idx, _ = find_col_index(cols, ["ki類型", "kitype"])
        t1_idx, _ = find_col_index(cols, ["標的1", "ticker1"])
        
        trade_date_idx, _ = find_col_index(cols, ["交易日"])
        issue_date_idx, _ = find_col_index(cols, ["發行日"])
        final_date_idx, _ = find_col_index(cols, ["最終", "評價"])
        maturity_date_idx, _ = find_col_index(cols, ["到期", "maturity"])
        tenure_idx, _ = find_col_index(cols, ["天期", "term", "tenure"])
        
        name_idx, _ = find_col_index(cols, ["理專", "姓名", "客戶"])
        line_id_idx, line_col_name = find_col_index(cols, ["line_id", "lineid", "lineuserid", "uid", "lind"])
        email_idx, _ = find_col_index(cols, ["email", "e-mail", "mail", "信箱"])

        if t1_idx is None:
            st.error("❌ 無法辨識「標的1」欄位，請檢查 Excel 表頭。")
            st.stop()

        clean_df = pd.DataFrame()
        clean_df['ID'] = df.iloc[:, id_idx]
        if name_idx is not None: clean_df['Name'] = df.iloc[:, name_idx].apply(clean_name_str)
        else: clean_df['Name'] = "貴賓"
        if line_id_idx is not None: clean_df['Line_ID'] = df.iloc[:, line_id_idx].astype(str).replace('nan', '').str.strip()
        else: clean_df['Line_ID'] = ""
        if email_idx is not None: clean_df['Email'] = df.iloc[:, email_idx].astype(str).replace('nan', '').str.strip()
        else: clean_df['Email'] = ""
        
        if type_idx is not None:
            clean_df['Product_Type'] = df.iloc[:, type_idx].astype(str).fillna("FCN")
        else:
            clean_df['Product_Type'] = "FCN"

        clean_df['TradeDate'] = pd.to_datetime(df.iloc[:, trade_date_idx], errors='coerce') if trade_date_idx else pd.NaT
        clean_df['IssueDate'] = pd.to_datetime(df.iloc[:, issue_date_idx], errors='coerce') if issue_date_idx else pd.Timestamp.min
        
        if maturity_date_idx: clean_df['MaturityDate'] = pd.to_datetime(df.iloc[:, maturity_date_idx], errors='coerce')
        else: clean_df['MaturityDate'] = pd.NaT
            
        clean_df['ValuationDate'] = pd.to_datetime(df.iloc[:, final_date_idx], errors='coerce') if final_date_idx else pd.NaT
        clean_df['TenureStr'] = df.iloc[:, tenure_idx] if tenure_idx else ""

        for idx, row in clean_df.iterrows():
            if pd.isna(row['MaturityDate']):
                calc_date = calculate_maturity(row, 'IssueDate', 'TenureStr')
                clean_df.at[idx, 'MaturityDate'] = calc_date
                if pd.isna(row['ValuationDate']): clean_df.at[idx, 'ValuationDate'] = calc_date

        def calc_tenure_display(row):
            if row['TenureStr'] != "": return str(row['TenureStr'])
            if pd.notna(row['MaturityDate']) and pd.notna(row['IssueDate']):
                days = (row['MaturityDate'] - row['IssueDate']).days
                return f"{int(round(days/30))}M" 
            return "-"
        clean_df['Tenure'] = clean_df.apply(calc_tenure_display, axis=1)

        clean_df['KO_Pct'] = df.iloc[:, ko_idx].apply(clean_percentage)
        clean_df['KI_Pct'] = df.iloc[:, ki_idx].apply(clean_percentage)
        clean_df['Strike_Pct'] = df.iloc[:, strike_idx].apply(clean_percentage) if strike_idx else 100.0
        
        clean_df['KO_Type'] = df.iloc[:, ko_type_idx] if ko_type_idx else "NC1" 
        clean_df['KI_Type'] = df.iloc[:, ki_type_idx] if ki_type_idx else "AKI"

        for i in range(1, 6):
            if i == 1: tx_idx = t1_idx
            else:
                tx_idx, _ = find_col_index(cols, [f"標的{i}"])
                if tx_idx is None: 
                    possible_idx = t1_idx + (i-1)*2
                    if possible_idx < len(df.columns): tx_idx = possible_idx
            
            if tx_idx is not None and tx_idx < len(df.columns):
                raw_ticker = df.iloc[:, tx_idx]
                clean_df[f'T{i}_Code'] = raw_ticker.apply(clean_ticker_symbol)
                
                if tx_idx + 1 < len(df.columns):
                    sample_val = df.iloc[0, tx_idx+1]
                    try:
                        float(sample_val)
                        clean_df[f'T{i}_Initial'] = pd.to_numeric(df.iloc[:, tx_idx + 1], errors='coerce').fillna(0)
                    except:
                        clean_df[f'T{i}_Initial'] = 0
                else:
                    clean_df[f'T{i}_Initial'] = 0
            else:
                clean_df[f'T{i}_Code'] = ""
                clean_df[f'T{i}_Initial'] = 0

        clean_df = clean_df.dropna(subset=['ID'])

        today_ts = pd.Timestamp(real_today)
        min_trade_date = clean_df['TradeDate'].min()
        
        if pd.isna(min_trade_date): start_download_date = today_ts - timedelta(days=30)
        else: start_download_date = min_trade_date - timedelta(days=7)

        all_tickers = []
        for i in range(1, 6):
            if f'T{i}_Code' in clean_df.columns:
                ts = clean_df[f'T{i}_Code'].dropna().unique().tolist()
                all_tickers.extend([t for t in ts if t != ""])
        all_tickers = list(set(all_tickers))

        if not all_tickers:
            st.error("❌ 找不到有效的標的代號。")
            st.stop()

        st.info(f"⏳ 下載美股資料... ({start_download_date.strftime('%Y-%m-%d')} ~ 今日)")
        
        try:
            history_data = yf.download(all_tickers, start=start_download_date, end=today_ts + timedelta(days=1))['Close']
        except Exception as e:
            st.error(f"美股連線失敗: {e}")
            st.stop()

        results = []
        individual_messages = [] 
        admin_summary_list = []
        lookback_date = today_ts - timedelta(days=lookback_days)

        for index, row in clean_df.iterrows():
            ko_thresh_val = row['KO_Pct'] if pd.notna(row['KO_Pct']) else 100.0
            ki_thresh_val = row['KI_Pct'] if pd.notna(row['KI_Pct']) else 60.0
            strike_thresh_val = row['Strike_Pct'] if pd.notna(row['Strike_Pct']) else 100.0
            
            ko_thresh = ko_thresh_val / 100.0
            ki_thresh = ki_thresh_val / 100.0
            strike_thresh = strike_thresh_val / 100.0
            nc_months = parse_nc_months(row['KO_Type'])
            nc_end_date = row['IssueDate'] + relativedelta(months=nc_months)
            
            is_dra = "DRA" in str(row['Product_Type']).upper()
            
            assets = []
            
            for i in range(1, 6):
                code = row.get(f'T{i}_Code', "")
                if code == "": continue
                initial = float(row.get(f'T{i}_Initial', 0))
                
                if initial == 0:
                    trade_date = row['TradeDate']
                    if pd.notna(trade_date):
                        try:
                            if len(all_tickers) == 1: s = history_data
                            else: s = history_data[code]
                            price_on_trade = s[s.index >= trade_date].head(1)
                            if not price_on_trade.empty:
                                initial = float(price_on_trade.iloc[0])
                        except: initial = 0
                
                if initial > 0:
                    assets.append({
                        'code': code, 'initial': initial, 'strike_price': initial * strike_thresh, 
                        'locked_ko': False, 'hit_ki': False, 'perf': 0.0, 'price': 0.0,
                        'ko_record': '', 'ki_record': ''
                    })
            
            if not assets: continue

            for asset in assets:
                try:
                    if len(all_tickers) == 1: s = history_data
                    else: s = history_data[asset['code']]
                    valid_s = s[s.index <= today_ts].dropna()
                    if not valid_s.empty:
                        curr = float(valid_s.iloc[-1])
                        asset['price'] = curr
                        asset['perf'] = curr / asset['initial']
                except: asset['price'] = 0

            product_status = "Running"
            early_redemption_date = None
            is_aki = "AKI" in str(row['KI_Type']).upper()

            if row['IssueDate'] <= today_ts:
                backtest_data = history_data[(history_data.index >= row['IssueDate']) & (history_data.index <= today_ts)]
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
                            
                            if is_aki and perf < ki_thresh and not asset['hit_ki']:
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

            locked_list = []; waiting_list = []; hit_ki_list = []; shadow_ko_list = []
            detail_cols = {}
            asset_detail_str = "" 
            any_below_strike_today = False
            dra_fail_list = []

            for i, asset in enumerate(assets):
                if asset['price'] > 0:
                    if not is_aki and asset['perf'] < ki_thresh: asset['hit_ki'] = True 
                    if is_dra and asset['perf'] < strike_thresh:
                        any_below_strike_today = True
                        dra_fail_list.append(asset['code'])

                if asset['locked_ko']: locked_list.append(asset['code'])
                else: waiting_list.append(asset['code'])
                if asset['hit_ki']: hit_ki_list.append(asset['code'])
                
                p_pct = round(asset['perf']*100, 2) if asset['price'] > 0 else 0.0
                status_icon = "✅" if asset['locked_ko'] else "⚠️" if asset['hit_ki'] else ""
                
                if is_dra and asset['price'] > 0:
                    if asset['perf'] < strike_thresh: status_icon += "🛑無息"
                    else: status_icon += "💸"

                price_display = round(asset['price'], 2) if asset['price'] > 0 else "N/A"
                initial_display = round(asset['initial'], 2)
                
                cell_text = f"【{asset['code']}】\n原: {initial_display}\n現: {price_display}\n({p_pct}%) {status_icon}"
                if asset['locked_ko']: cell_text += f"\nKO {asset['ko_record']}"
                if asset['hit_ki']: cell_text += f"\nKI {asset['ki_record']}"
                detail_cols[f"T{i+1}_Detail"] = cell_text
                
                asset_detail_str += f"{asset['code']}: {p_pct}% {status_icon} (原:{initial_display})\n"

            hit_any_ki = any(a['hit_ki'] for a in assets)
            all_above_strike_now = all((a['perf'] >= strike_thresh if a['price'] > 0 else False) for a in assets)
            
            valid_assets = [a for a in assets if a['perf'] > 0]
            if valid_assets:
                worst_asset = min(valid_assets, key=lambda x: x['perf'])
                worst_perf = worst_asset['perf']
            else:
                worst_perf = 0
            
            # 🌟 狀態文字產生 (KI 優先)
            status_msgs = []
            line_status_short = "" 
            need_notify = False

            if today_ts < row['IssueDate']:
                status_msgs.append("⏳ 未發行")
            elif product_status == "Early Redemption":
                status_msgs.append(f"🎉 提前出場 ({early_redemption_date.strftime('%Y-%m-%d')})")
                if early_redemption_date >= lookback_date:
                    line_status_short = "🎉 恭喜！已提前出場 (KO)"
                    need_notify = True
            elif pd.notna(row['ValuationDate']) and today_ts >= row['ValuationDate']:
                if all_above_strike_now:
                     status_msgs.append("💰 到期獲利")
                     line_status_short = "💰 到期獲利"
                elif hit_any_ki:
                     status_msgs.append("😭 到期接股")
                     line_status_short = "😭 到期接股"
                else:
                     status_msgs.append("🛡️ 到期保本")
                     line_status_short = "🛡️ 到期保本"
                if row['ValuationDate'] >= lookback_date: need_notify = True
            else:
                # 執行中
                if today_ts < nc_end_date:
                    status_msgs.append("🔒 NC閉鎖期")
                else:
                    status_msgs.append("👀 比價中")
                
                # 1. KI 檢查 (絕對優先，置頂顯示)
                if hit_any_ki:
                    status_msgs.insert(0, f"☠️ 已跌破KI ({','.join(hit_ki_list)})")
                    line_status_short = f"⚠️ 警告：已跌破 KI ({','.join(hit_ki_list)})"
                    need_notify = True # 只要破 KI，強制通知
                
                # 2. DRA 檢查
                if is_dra:
                    if any_below_strike_today:
                        status_msgs.append(f"🛑 DRA暫停計息 ({','.join(dra_fail_list)})")
                        if notify_ki_daily:
                            # 如果 KI 已經通知了，這裡附加就好，不要覆蓋 KI 的嚴重性
                            if not line_status_short:
                                line_status_short = f"🛑 DRA 暫停計息 ({','.join(dra_fail_list)} 跌破)"
                            else:
                                line_status_short += f" & 🛑 DRA 暫停計息"
                            need_notify = True
                    else:
                        status_msgs.append("💸 DRA計息中")

            final_status = "\n".join(status_msgs)

            if line_status_short:
                admin_summary_list.append(f"● {row['ID']} ({row['Name']}): {line_status_short}")

            line_ids = [x.strip() for x in re.split(r'[;,，]', str(row.get('Line_ID', ''))) if x.strip()]
            emails = [x.strip() for x in re.split(r'[;,，]', str(row.get('Email', ''))) if x.strip()]
            
            mat_date_str = row['MaturityDate'].strftime('%Y-%m-%d') if pd.notna(row['MaturityDate']) else "-"
            common_msg_body = (
                f"Hi {row['Name']} 您好，\n"
                f"您的結構型商品 {row['ID']} ({row['Product_Type']}) 最新狀態：\n\n"
                f"【{line_status_short}】\n\n"
                f"{asset_detail_str}"
                f"📅 到期日: {mat_date_str}\n"
                f"------------------\n"
                f"貼心通知"
            )

            if need_notify and line_status_short:
                for uid in line_ids:
                    if uid.startswith("U") or uid.startswith("C"):
                        individual_messages.append({'target': uid, 'msg': common_msg_body})
                for mail in emails:
                    if "@" in mail:
                        subject = f"【ELN通知】{row['ID']} 最新狀態"
                        mail_body = common_msg_body + "\n(本信件由系統自動發送)"
                        individual_messages.append({'target': mail, 'subj': subject, 'msg': mail_body})

            row_res = {
                "債券代號": row['ID'], "Name": row['Name'], "Type": row['Product_Type'],
                "狀態": final_status, "最差表現": f"{round(worst_perf*100, 2)}%",
                "交易日": row['TradeDate'].strftime('%Y-%m-%d') if pd.notna(row['TradeDate']) else "-",
                "NC月份": f"{nc_months}M",
            }
            row_res.update(detail_cols)
            results.append(row_res)

        if not results:
            st.warning("⚠️ 無資料")
        else:
            final_df = pd.DataFrame(results)
            
            def color_status(val):
                s = str(val)
                # 順序很重要，先判斷壞消息
                if "跌破KI" in s or "接股" in s: return 'background-color: #f8d7da; color: red; font-weight: bold'
                if "暫停" in s: return 'background-color: #fff3cd; color: #856404'
                if "提前" in s or "獲利" in s or "計息中" in s: return 'background-color: #d4edda; color: green'
                return ''

            t_cols = [c for c in final_df.columns if '_Detail' in c]; t_cols.sort()
            display_cols = ['債券代號', 'Type', 'Name', '狀態', '最差表現'] + t_cols + ['交易日']
            
            st.subheader("📋 監控列表")
            st.dataframe(final_df[display_cols].style.applymap(color_status, subset=['狀態']), height=600, use_container_width=True)

            st.markdown("### 📢 發送操作")
            
            if st.session_state['is_sent']:
                st.success("✅ 發送完成！")
                if st.button("🔄 重置"):
                    st.session_state['is_sent'] = False
                    st.rerun()
            else:
                count = len(individual_messages)
                btn_label = f"🚀 發送 LINE 通知 (預計: {count} 則)"
                
                if st.button(btn_label, type="primary"):
                    
                    if admin_summary_list and MY_LINE_USER_ID:
                        summary_text = f"【ELN 戰情快報】\n📅 {real_today.strftime('%Y/%m/%d')}\n----------------\n" + "\n".join(admin_summary_list)
                        if count > 0: summary_text += f"\n\n(系統將繼續發送 {count} 則客戶通知...)"
                        else: summary_text += f"\n\n(今日無須發送客戶通知)"
                        
                        send_line_push(MY_LINE_USER_ID, summary_text)
                        st.toast("✅ 管理員摘要已發送", icon="📢")

                    success_cnt = 0
                    bar = st.progress(0, text="正在發送客戶通知...")
                    
                    for idx, item in enumerate(individual_messages):
                        if send_line_push(item['target'], item['msg']):
                            success_cnt += 1
                        bar.progress((idx+1)/count)
                    
                    bar.empty()

                    st.session_state['is_sent'] = True
                    st.success(f"🎉 成功發送 {success_cnt} 則通知！")
                    st.balloons()

    except Exception as e:
        st.error(f"發生錯誤：{e}")
