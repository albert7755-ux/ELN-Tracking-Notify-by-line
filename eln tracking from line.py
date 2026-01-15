import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import re
from dateutil.relativedelta import relativedelta
from linebot import LineBotApi
from linebot.models import TextSendMessage

# --- 設定網頁 ---
st.set_page_config(page_title="ELN 戰情室 (LINE 多人發送版)", layout="wide")

# ==========================================
# 🔐 雲端機密讀取
# ==========================================
try:
    LINE_ACCESS_TOKEN = st.secrets["LINE_ACCESS_TOKEN"]
    MY_LINE_USER_ID = st.secrets["MY_LINE_USER_ID"]
except Exception:
    st.error("⚠️ 尚未設定 Secrets！")
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
        st.success(f"✅ LINE 連線成功")
    else:
        st.error("❌ LINE 設定未完成")

    st.markdown("---")
    real_today = datetime.now()
    st.info(f"📅 今天日期：{real_today.strftime('%Y-%m-%d')}")
    st.caption("鎖定真實日期")
    
    st.markdown("---")
    st.info("💡 **多人發送技巧**\nExcel 的 ID 欄位可以用「逗號」分隔多個人。\n例如: `U123..., U456...`")

# --- 函數區 ---
def send_line_push(target_user_id, message_text):
    if not LINE_ACCESS_TOKEN or not target_user_id:
        return False
    try:
        uid = str(target_user_id).strip()
        if not uid.startswith("U") or len(uid) < 10:
            return False
            
        line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
        line_bot_api.push_message(uid, TextSendMessage(text=message_text))
        return True
    except Exception as e:
        print(f"發送失敗 ({target_user_id}): {e}")
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

def clean_name_str(val):
    if pd.isna(val): return "貴賓"
    s = str(val).strip()
    if s.lower() == 'nan' or s == "": return "貴賓"
    return s

def find_col_index(columns, include_keywords, exclude_keywords=None):
    for idx, col_name in enumerate(columns):
        col_str = str(col_name).strip().lower()
        if exclude_keywords:
            if any(ex in col_str for ex in exclude_keywords): continue
        if any(inc in col_str for inc in include_keywords):
            return idx, col_name
    return None, None

# --- 主畫面 ---
st.title("📊 ELN 結構型商品 - LINE 多人發送版")

uploaded_file = st.file_uploader("請上傳 Excel (支援多組 ID 用逗號分隔)", type=['xlsx', 'csv'], key="uploader")

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
        
        id_idx, _ = find_col_index(cols, ["債券", "代號", "id"]) or (0, "")
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
        name_idx, _ = find_col_index(cols, ["理專", "姓名", "客戶"])
        
        line_id_idx, line_col_name = find_col_index(cols, ["line_id", "lineid", "line user id", "uid", "lind", "lind_id"])

        if line_id_idx is not None:
            st.toast(f"✅ ID 欄位：{line_col_name} (支援逗號分隔)", icon="👥")

        if t1_idx is None or ko_idx is None:
            st.error("❌ 嚴重錯誤：無法辨識關鍵欄位。")
            st.stop()

        clean_df = pd.DataFrame()
        clean_df['ID'] = df.iloc[:, id_idx]
        
        if name_idx is not None:
            clean_df['Name'] = df.iloc[:, name_idx].apply(clean_name_str)
        else:
            clean_df['Name'] = "貴賓"
            
        if line_id_idx is not None:
            clean_df['Line_ID'] = df.iloc[:, line_id_idx].astype(str).replace('nan', '').str.strip()
        else:
            clean_df['Line_ID'] = ""

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
        
        # 下載股價
        today_ts = pd.Timestamp(real_today)
        min_issue_date = clean_df['IssueDate'].min()
        start_date = today_ts - timedelta(days=30) if pd.isna(min_issue_date) else min(min_issue_date, today_ts - timedelta(days=14))
            
        st.info(f"下載美股資料... (基準日: {real_today.strftime('%Y-%m-%d')}) ☕")
        
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

        # 運算邏輯
        results = []
        admin_summary_list = [] 
        individual_messages = [] 

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

            locked_list = []; waiting_list = []; hit_ki_list = []; shadow_ko_list = []
            detail_cols = {}
            asset_detail_str = "" 

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
                
                asset_detail_str += f"{asset['code']}: {p_pct}% {status_icon}\n"

            hit_any_ki = any(a['hit_ki'] for a in assets)
            all_above_strike_now = all((a['perf'] >= strike_thresh if a['price'] > 0 else False) for a in assets)
            
            valid_assets = [a for a in assets if a['perf'] > 0]
            if valid_assets:
                worst_asset = min(valid_assets, key=lambda x: x['perf'])
                worst_perf = worst_asset['perf']
                worst_code = worst_asset['code']
            else:
                worst_perf = 0; worst_code = "N/A"
            
            final_status = ""
            line_status_short = "" 

            if today_ts < row['IssueDate']:
                final_status = "⏳ 未發行"
            elif product_status == "Early Redemption":
                final_status = f"🎉 提前出場\n({early_redemption_date.strftime('%Y-%m-%d')})"
                line_status_short = "🎉 恭喜！已提前出場 (KO)"
            elif pd.notna(row['ValuationDate']) and today_ts >= row['ValuationDate']:
                if all_above_strike_now:
                     final_status = "💰 到期獲利\n(全數 > 執行價)"
                     line_status_short = "💰 到期獲利"
                elif hit_any_ki:
                     final_status = f"😭 到期接股"
                     line_status_short = f"😭 到期接股 (Worst: {worst_code})"
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
                    line_status_short = f"⚠️ 注意：KI 已跌破 ({','.join(hit_ki_list)})"

            # 收集摘要
            if line_status_short:
                admin_summary_list.append(f"● {row['ID']} ({row['Name']}): {line_status_short}")
            
            # 🚀 多人發送邏輯 (LINE)
            target_ids = row.get('Line_ID', '')
            # 使用正則表達式，同時支援 逗號, 分號; 全形逗號，
            id_list = [x.strip() for x in re.split(r'[;,，]', str(target_ids)) if x.strip()]

            if id_list and line_status_short:
                msg = (f"Hi {row['Name']} 您好，\n"
                       f"您的結構型商品 {row['ID']} 最新狀態：\n\n"
                       f"【{line_status_short}】\n\n"
                       f"{asset_detail_str}"
                       f"📅 到期日: {mat_date_str}\n"
                       f"------------------\n"
                       f"貼心通知")
                
                # 每個 ID 都加進去發送列表
                for uid in id_list:
                    if uid.startswith("U"):
                        individual_messages.append( (uid, msg) )

            trade_date_str = row['TradeDate'].strftime('%Y-%m-%d') if pd.notna(row['TradeDate']) else "-"
            issue_date_str = row['IssueDate'].strftime('%Y-%m-%d') if pd.notna(row['IssueDate']) else "-"
            val_date_str = row['ValuationDate'].strftime('%Y-%m-%d') if pd.notna(row['ValuationDate']) else "-"
            mat_date_str = row['MaturityDate'].strftime('%Y-%m-%d') if pd.notna(row['MaturityDate']) else "-"

            row_res = {
                "債券代號": row['ID'], "Line_ID": target_ids, "天期": row['Tenure'], "收件人": row['Name'],
                "狀態": final_status, "最差表現": f"{round(worst_perf*100, 2)}%",
                "KO設定": f"{ko_thresh_val}%", "KI設定": f"{ki_thresh_val}%", "執行價": f"{strike_thresh_val}%",
                "交易日": trade_date_str, "發行日": issue_date_str, "最終評價": val_date_str, "到期日": mat_date_str
            }
            row_res.update(detail_cols)
            results.append(row_res)

        # 顯示結果
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
            display_cols = ['債券代號', '天期', '狀態', '最差表現'] + t_cols + ['Line_ID', 'KO設定', 'KI設定', '執行價', '交易日', '發行日', '最終評價', '到期日']
            column_config = {
                "狀態": st.column_config.TextColumn("目前狀態摘要", width="large"),
                "Line_ID": st.column_config.TextColumn("Line IDs", width="medium"),
                "債券代號": st.column_config.TextColumn("代號", width="small"),
                "天期": st.column_config.TextColumn("天期", width="small"),
                "KO設定": st.column_config.TextColumn("KO", width="small"),
                "KI設定": st.column_config.TextColumn("KI", width="small"),
                "執行價": st.column_config.TextColumn("Strike", width="small"),
                "最差表現": st.column_config.TextColumn("Worst Of", width="small"),
            }
            for i, c in enumerate(t_cols): column_config[c] = st.column_config.TextColumn(f"標的 {i+1} (原始/現價/狀態)", width="large")

            st.dataframe(final_df[display_cols].style.applymap(color_status, subset=['狀態']), use_container_width=True, column_config=column_config, height=600, hide_index=True)
            
            # 按鈕操作
            st.markdown("### 📢 發送操作")
            
            if st.session_state['is_sent']:
                st.success("✅ 本次檔案已發送完成！")
                if st.button("🔄 重置狀態 (讓我再發一次)"):
                    st.session_state['is_sent'] = False
                    st.rerun()
            else:
                btn_label = f"📲 發送 LINE 通知 (預計: {len(individual_messages)} 位收件者 + 1 位管理員)"
                if st.button(btn_label, type="primary"):
                    success_count = 0
                    
                    # 發給個別客戶
                    progress_text = "正在發送客戶通知..."
                    my_bar = st.progress(0, text=progress_text)
                    
                    total_msgs = len(individual_messages)
                    for idx, (uid, msg) in enumerate(individual_messages):
                        if send_line_push(uid, msg):
                            success_count += 1
                        if total_msgs > 0:
                            my_bar.progress((idx + 1) / total_msgs, text=f"發送中... ({idx+1}/{total_msgs})")
                    
                    my_bar.empty()
                    
                    # 發給管理員
                    if admin_summary_list:
                        admin_msg = f"【ELN 戰情快報】\n📅 {real_today.strftime('%Y/%m/%d')}\n----------------\n" + "\n".join(admin_summary_list)
                        if success_count > 0:
                            admin_msg += f"\n\n(已另行發送 {success_count} 則個別通知)"
                        send_line_push(MY_LINE_USER_ID, admin_msg)
                    else:
                        send_line_push(MY_LINE_USER_ID, f"【ELN 戰情快報】\n📅 {real_today.strftime('%Y/%m/%d')}\n今日無特殊事件。")
                    
                    st.session_state['is_sent'] = True
                    st.success(f"🎉 發送完畢！成功發送 {success_count} 則客戶通知。")
                    st.balloons()

    except Exception as e:
        st.error(f"發生錯誤：{e}")
else:
    st.info("👆 請上傳 Excel (含 Line_ID 欄位)")
