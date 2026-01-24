# @title 🚀 TW-PocketScreener V2.3 (大師導覽版)
# @markdown 🎓 **新增：整合「黃金投資法則」歡迎教育頁面。**
# @markdown 🔄 **機制：單一 HTML 檔案，前端切換「教學模式」與「選股模式」。**
# @markdown 🏆 **功能：完整保留 V2.2 所有篩選功能與 V2.2.1 的語法修正。**
# @markdown ⏳ **預計耗時：約 40~60 分鐘 (需抓取全台股數據)。**

import subprocess
import sys
import json
import time
import requests
import io
import pandas as pd
import numpy as np
import concurrent.futures
import warnings
import random
import logging
from datetime import datetime, timedelta

# --- 0. 環境準備 ---
def install(package):
    try:
        __import__(package)
    except ImportError:
        print(f"📦 正在安裝 {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('yfinance')
install('pandas')
install('numpy')
install('lxml')
install('requests')

try:
    from fake_useragent import UserAgent
except ImportError:
    print("📦 正在安裝 fake-useragent...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fake-useragent"])
    from fake_useragent import UserAgent

import yfinance as yf

# 1. 設定環境
warnings.simplefilter(action='ignore', category=FutureWarning)
yf_logger = logging.getLogger('yfinance')
yf_logger.setLevel(logging.CRITICAL)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj): return None
            return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, np.bool_): return bool(obj)
        else: return super(NpEncoder, self).default(obj)

# ==========================================
# 1. 取得全台股清單
# ==========================================
tw_time = datetime.utcnow() + timedelta(hours=8)
print(f"📥 [1/4] 正在獲取全台股清單 ({tw_time.strftime('%H:%M:%S')})...")

def get_tw_stock_list():
    stock_list = []
    ua = UserAgent()
    
    def fetch_isin(url, suffix):
        try:
            headers = {'User-Agent': ua.random}
            r = requests.get(url, headers=headers)
            r.encoding = 'big5'
            dfs = pd.read_html(io.StringIO(r.text))
            df = max(dfs, key=lambda d: d.shape[0])
            for index, row in df.iterrows():
                try:
                    col0 = str(row.iloc[0]).strip()
                    parts = col0.split()
                    if len(parts) >= 2:
                        code = parts[0]
                        name = parts[1]
                        if len(code) == 4 and code.isdigit():
                            stock_list.append({'id': code, 'name': name, 'suffix': suffix, 'ticker': code + suffix})
                except: continue
        except: pass
    
    fetch_isin("https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", ".TW")
    fetch_isin("https://isin.twse.com.tw/isin/C_public.jsp?strMode=4", ".TWO")
    return stock_list

all_stocks = get_tw_stock_list()
if not all_stocks: 
    print("⚠️ 清單抓取失敗，使用測試模式。")
    all_stocks = [{'id': '2330', 'name': '台積電', 'suffix': '.TW', 'ticker': '2330.TW'}]
print(f"📋 共取得 {len(all_stocks)} 檔股票。")

# ==========================================
# 2. 批次下載股價
# ==========================================
print("\n📥 [2/4] 啟動批次股價下載 (Chunk Size: 100)...")

processed_data = {}
BATCH_SIZE = 100
chunks = [all_stocks[i:i + BATCH_SIZE] for i in range(0, len(all_stocks), BATCH_SIZE)]
total_batches = len(chunks)

for i, chunk in enumerate(chunks):
    tickers = [s['ticker'] for s in chunk]
    sys.stdout.write(f"\r   - 批次 {i+1}/{total_batches} (已成功: {len(processed_data)} 檔)   ")
    sys.stdout.flush()
    
    try:
        data = yf.download(tickers, period="3mo", group_by='ticker', auto_adjust=True, threads=True, progress=False)
        for stock in chunk:
            t = stock['ticker']
            try:
                if len(tickers) == 1: df = data
                else:
                    if t not in data.columns.levels[0]: continue
                    df = data[t]
                
                if df.empty or 'Close' not in df.columns or df['Close'].isnull().all(): continue
                
                close = df['Close'].dropna().tolist()
                if len(close) < 2: continue
                
                vol = 0
                if 'Volume' in df.columns: vol = int(df['Volume'].tail(5).mean() / 1000)
                
                if vol < 5: continue 

                price = round(close[-1], 2)
                ma20 = sum(close[-20:]) / 20 if len(close) >= 20 else 0
                
                processed_data[t] = {
                    "id": stock['id'], "name": stock['name'],
                    "price": price, "vol": vol,
                    "sparkline": [round(x, 2) for x in close], 
                    "ma_bull": price > ma20,
                    # 指標初始化
                    "eps_ttm": 0, "eps_avg": 0, 
                    "roe_ttm": 0, "roe_avg": 0, "roa": 0,
                    "gross_margin": 0, "op_margin": 0, 
                    "pe": 0, "pb": 0, "yield": 0, "yield_avg": 0,
                    "rev_growth": 0, "cons_div": 0,
                    "core_purity": 0, 
                    "gm_stability": 999, 
                    "payout_ratio": 0, 
                    "tags": [] 
                }
            except: continue
    except: pass

print(f"\n✅ 股價獲取完成！有效: {len(processed_data)} 檔")

# ==========================================
# 3. 深層挖掘財報
# ==========================================
print("\n📥 [3/4] 正在深層挖掘財報數據 (含V2.2新增濾鏡)...")
print("   ⚠️ 需計算3年毛利變動與本業比重，預計需 40~60 分鐘。")

def fetch_deep_stats(ticker):
    time.sleep(random.uniform(1.0, 3.0))
    
    try:
        stock = yf.Ticker(ticker)
        try:
            info = stock.info
        except:
            time.sleep(2)
            stock = yf.Ticker(ticker)
            info = stock.info

        pe = round(info.get('trailingPE', 0), 2)
        pb = round(info.get('priceToBook', 0), 2)
        eps_ttm = info.get('trailingEps', 0)
        roe_ttm = round(info.get('returnOnEquity', 0) * 100, 2)
        roa = round(info.get('returnOnAssets', 0) * 100, 2)
        gross_margin = round(info.get('grossMargins', 0) * 100, 2)
        op_margin = round(info.get('operatingMargins', 0) * 100, 2)
        rev_growth = round(info.get('revenueGrowth', 0) * 100, 2)
        payout_ratio = round(info.get('payoutRatio', 0) * 100, 2) if info.get('payoutRatio') else 0

        div_yield = 0
        if info.get('dividendRate') and info.get('regularMarketPrice'):
             div_yield = round((info['dividendRate'] / info['regularMarketPrice']) * 100, 2)
        
        yield_avg = info.get('fiveYearAvgDividendYield', 0)
        if yield_avg is None: yield_avg = 0
        else: yield_avg = round(yield_avg, 2)

        income = pd.DataFrame()
        try: income = stock.income_stmt
        except: pass

        eps_avg = 0
        if not income.empty:
            try:
                if 'Basic EPS' in income.index:
                    eps_series = income.loc['Basic EPS'].head(5).dropna()
                    if len(eps_series) > 0: eps_avg = round(eps_series.mean(), 2)
                elif 'Diluted EPS' in income.index:
                    eps_series = income.loc['Diluted EPS'].head(5).dropna()
                    if len(eps_series) > 0: eps_avg = round(eps_series.mean(), 2)
            except: eps_avg = eps_ttm
        else: eps_avg = eps_ttm

        core_purity = 0
        if not income.empty:
            try:
                op_inc = income.loc['Operating Income'].iloc[0]
                pretax = income.loc['Pretax Income'].iloc[0]
                if pretax > 0: core_purity = round((op_inc / pretax) * 100, 2)
            except: pass

        gm_stability = 999
        if not income.empty:
            try:
                gp_rows = income.loc['Gross Profit'].head(3)
                rev_rows = income.loc['Total Revenue'].head(3)
                if len(gp_rows) >= 3 and len(rev_rows) >= 3:
                    margins = []
                    for i in range(3):
                        if rev_rows.iloc[i] > 0: margins.append((gp_rows.iloc[i] / rev_rows.iloc[i]) * 100)
                    if len(margins) == 3: gm_stability = round(max(margins) - min(margins), 2)
            except: pass

        roe_avg = 0
        try:
            bs = stock.balance_sheet
            if not bs.empty and not income.empty:
                ni = income.loc['Net Income']
                eq_key = next((k for k in bs.index if 'Stockholders Equity' in k or 'Total Equity' in k), None)
                if eq_key:
                    eq = bs.loc[eq_key]
                    roe_series = (ni / eq) * 100
                    recent_roe = roe_series.head(5).dropna()
                    if len(recent_roe) > 0: roe_avg = round(recent_roe.mean(), 2)
        except: pass
        if roe_avg == 0: roe_avg = roe_ttm

        cons_div = 0
        try:
            divs = stock.history(period="15y")['Dividends']
            if not divs.empty:
                yearly_divs = divs.groupby(divs.index.year).sum()
                current_y = datetime.now().year
                check_year = current_y - 1
                if check_year not in yearly_divs.index or yearly_divs.loc[check_year] == 0:
                    if (check_year - 1) in yearly_divs.index and yearly_divs.loc[check_year - 1] > 0:
                        check_year -= 1
                while check_year in yearly_divs.index and yearly_divs.loc[check_year] > 0:
                    cons_div += 1
                    check_year -= 1
        except:
            if div_yield > 0: cons_div = 1

        return {
            "pe": pe, "pb": pb, "yield": div_yield, "yield_avg": yield_avg,
            "eps_ttm": eps_ttm, "eps_avg": eps_avg, 
            "roe_ttm": roe_ttm, "roe_avg": roe_avg, 
            "roa": roa, "gross_margin": gross_margin, "op_margin": op_margin,
            "rev_growth": rev_growth, "cons_div": cons_div,
            "core_purity": core_purity, "gm_stability": gm_stability, "payout_ratio": payout_ratio
        }
    except:
        return None

tickers_to_enrich = list(processed_data.keys())
enriched_count = 0
count = 0
total = len(tickers_to_enrich)
MAX_WORKERS = 2 
start_time = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_ticker = {executor.submit(fetch_deep_stats, t): t for t in tickers_to_enrich}
    
    for future in concurrent.futures.as_completed(future_to_ticker):
        t = future_to_ticker[future]
        count += 1
        
        if count % 50 == 0: time.sleep(10)
        
        if count % 5 == 0 or count == total:
            elapsed = time.time() - start_time
            avg_time = elapsed / count
            remain = (total - count) * avg_time / 60
            sys.stdout.write(f"\r   - 進度: {count}/{total} ({count/total*100:.1f}%) | 成功: {enriched_count} | 剩餘: ~{remain:.0f}分   ")
            sys.stdout.flush()
            
        try:
            stats = future.result()
            if stats:
                processed_data[t].update(stats)
                
                tags = []
                is_golden = (processed_data[t]['eps_ttm'] >= 1 and 
                             processed_data[t]['eps_avg'] >= 2 and
                             processed_data[t]['yield_avg'] >= 5 and
                             processed_data[t]['cons_div'] >= 10 and
                             processed_data[t]['roe_avg'] >= 15 and
                             processed_data[t]['core_purity'] >= 80 and
                             processed_data[t]['gm_stability'] <= 5 and
                             processed_data[t]['payout_ratio'] >= 60 and processed_data[t]['payout_ratio'] <= 100)
                             
                if is_golden: tags.append("🏆黃金存股")
                if processed_data[t]['yield'] > 5: tags.append("💰高殖利")
                if processed_data[t]['roe_avg'] > 15: tags.append("🔥高ROE")
                if processed_data[t]['ma_bull']: tags.append("📈站上月線")
                
                processed_data[t]['tags'] = tags
                enriched_count += 1
        except: pass

print(f"\n\n✅ 深度分析完成。成功獲取完整數據: {enriched_count}/{len(processed_data)} 檔")

# 轉 JSON
final_db = list(processed_data.values())
try:
    json_db = json.dumps(final_db, cls=NpEncoder, ensure_ascii=False)
except Exception as e:
    print(f"JSON Error: {e}"); json_db = "[]"

# --- 最終統計報告 ---
print("\n" + "="*35)
print("📊 TW-PocketScreener V2.3 執行報告")
print("="*35)
print(f"📋 監測總數 : {len(all_stocks)} 檔")
print(f"✅ 股價有效 : {len(processed_data)} 檔")
print(f"💎 財報完整 : {enriched_count} 檔")
print("="*35 + "\n")

# ==========================================
# 4. 生成 HTML (V2.3: 雙視圖整合版)
# ==========================================
update_time = (datetime.utcnow() + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M')

html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TW-PocketScreener V2.3 - 存股大師版</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/alpinejs/3.13.3/cdn.min.js" defer></script>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Noto Sans TC', sans-serif; -webkit-tap-highlight-color: transparent; }}
        .animate-fade-in {{ animation: fadeIn 0.5s ease-out; }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        .no-scrollbar::-webkit-scrollbar {{ display: none; }}
        .no-scrollbar {{ -ms-overflow-style: none; scrollbar-width: none; }}
        [x-cloak] {{ display: none !important; }}
    </style>
</head>
<body class="bg-slate-50 text-slate-800 h-screen flex flex-col overflow-hidden">

    <div id="welcome-view" class="flex-1 overflow-y-auto">
        <div id="modal-overlay" class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/70 backdrop-blur-sm hidden opacity-0 transition-opacity duration-300">
            <div class="bg-white rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden flex flex-col max-h-[90vh] transform scale-95 transition-transform duration-300" id="modal-container">
                <div class="bg-slate-800 text-white p-6 relative">
                    <button onclick="closeModal()" class="absolute top-4 right-4 text-slate-400 hover:text-white transition-colors"><i data-lucide="x"></i></button>
                    <div class="flex items-center gap-4"><span id="modal-icon" class="text-4xl"></span><div><h3 id="modal-title" class="text-xl font-bold"></h3><p id="modal-subtitle" class="text-yellow-400 text-sm font-medium"></p></div></div>
                </div>
                <div class="p-6 overflow-y-auto">
                    <div id="modal-body" class="space-y-4 text-slate-700 leading-relaxed text-sm md:text-base"></div>
                    <div class="mt-8 bg-blue-50 p-4 rounded-xl border border-blue-100"><h4 class="text-blue-800 font-bold text-sm mb-2 flex items-center gap-2"><i data-lucide="lightbulb" size="16"></i> 重點筆記</h4><p id="modal-highlight" class="text-blue-700 text-sm"></p></div>
                </div>
                <div class="p-4 border-t border-slate-100 bg-slate-50 flex gap-3">
                    <button id="modal-search-btn" class="flex-1 bg-white border border-slate-300 text-slate-700 py-2.5 rounded-lg text-sm font-bold hover:bg-slate-50 transition-colors flex items-center justify-center gap-2"><i data-lucide="search" size="16"></i> Google 搜尋更多</button>
                    <button onclick="closeModal()" class="flex-1 bg-slate-800 text-white py-2.5 rounded-lg text-sm font-bold hover:bg-slate-700 transition-colors">我瞭解了</button>
                </div>
            </div>
        </div>

        <header class="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white py-20 px-4 relative overflow-hidden">
            <div class="absolute top-0 right-0 p-10 opacity-10 animate-pulse"><i data-lucide="trending-up" width="400" height="400"></i></div>
            <div class="max-w-5xl mx-auto relative z-10 text-center">
                <div class="inline-block bg-yellow-500 text-slate-900 font-bold px-4 py-1 rounded-full text-sm mb-4">財富自由的必修課</div>
                <h1 class="text-4xl md:text-6xl font-bold mb-6 text-yellow-400 tracking-tight">黃金投資法則</h1>
                <p class="text-xl md:text-2xl text-slate-300 mb-10 max-w-2xl mx-auto leading-relaxed">站在巨人的肩膀上，融合 <span class="text-white font-semibold">巴菲特價值投資</span> 與 <span class="text-white font-semibold">台灣名家存股心法</span>，<br/>打造穿越牛熊的穩健致富策略。</p>
                <div class="flex flex-col sm:flex-row justify-center gap-4">
                    <button onclick="scrollToSection('core-content')" class="bg-slate-700 hover:bg-slate-600 text-white font-bold py-4 px-8 rounded-full transition duration-300 shadow-xl flex items-center justify-center gap-2 text-lg border border-slate-600">探索投資心法</button>
                    <button onclick="enterScreener()" class="bg-red-600 hover:bg-red-500 text-white font-bold py-4 px-8 rounded-full transition duration-300 shadow-xl flex items-center justify-center gap-2 text-lg group animate-pulse"><i data-lucide="rocket"></i> 立即使用選股工具</button>
                </div>
            </div>
        </header>

        <div id="core-content" class="max-w-6xl mx-auto px-4 py-16">
            <div class="flex flex-wrap justify-center mb-10 gap-2">
                <button onclick="switchTab('buffett')" id="btn-buffett" class="tab-btn px-6 py-3 rounded-full font-bold transition-all flex items-center gap-2 bg-slate-800 text-yellow-400 shadow-lg scale-105"><i data-lucide="shield" size="18"></i> 巴菲特與國際大師</button>
                <button onclick="switchTab('taiwan')" id="btn-taiwan" class="tab-btn px-6 py-3 rounded-full font-bold transition-all flex items-center gap-2 bg-white text-slate-600 hover:bg-slate-100"><i data-lucide="users" size="18"></i> 台灣存股名家</button>
                <button onclick="switchTab('golden')" id="btn-golden" class="tab-btn px-6 py-3 rounded-full font-bold transition-all flex items-center gap-2 bg-white text-slate-600 hover:bg-slate-100"><i data-lucide="lightbulb" size="18"></i> 黃金法則總結</button>
            </div>
            <div class="bg-white rounded-3xl p-6 md:p-10 shadow-lg border border-slate-100 min-h-[500px]">
                <div id="tab-buffett" class="tab-content animate-fade-in space-y-12">
                    <div class="flex flex-col md:flex-row items-start gap-8">
                        <div class="flex-1">
                            <div class="flex items-center gap-3 mb-4"><div class="w-16 h-16 rounded-full bg-slate-200 overflow-hidden flex items-center justify-center text-4xl shadow-inner">👑</div><h3 class="text-3xl font-bold text-slate-800">華倫·巴菲特 (Warren Buffett)</h3></div>
                            <p class="text-slate-600 text-lg leading-relaxed mb-6">巴菲特被譽為「奧馬哈的神諭」，他透過波克夏·海瑟威公司創造了史上最驚人的複利奇蹟。他的策略不僅是投資股票，更是<b>「購買企業的一部分」</b>。</p>
                            <div class="bg-slate-50 border-l-4 border-yellow-500 p-6 rounded-r-xl mb-6 shadow-sm"><i data-lucide="quote" class="text-yellow-500 mb-2"></i><p class="text-xl font-serif text-slate-800 italic mb-2">"Price is what you pay. Value is what you get."</p><p class="text-slate-600 font-medium">—— 價格是你付出的，價值是你得到的。</p></div>
                        </div>
                        <div class="w-full md:w-1/3 bg-slate-800 text-yellow-400 p-6 rounded-2xl shadow-lg">
                            <h4 class="font-bold text-xl mb-4 border-b border-slate-600 pb-2">價值投資鐵三角</h4>
                            <ul class="space-y-4">
                                <li class="flex items-start gap-3"><div class="bg-yellow-500 text-slate-900 rounded-full w-6 h-6 flex items-center justify-center font-bold flex-shrink-0">1</div><div><strong class="block text-white">經濟護城河 (Moat)</strong><span class="text-sm text-slate-300">競爭者難以跨越的優勢（如品牌、專利）。</span></div></li>
                                <li class="flex items-start gap-3"><div class="bg-yellow-500 text-slate-900 rounded-full w-6 h-6 flex items-center justify-center font-bold flex-shrink-0">2</div><div><strong class="block text-white">安全邊際 (Margin of Safety)</strong><span class="text-sm text-slate-300">用 0.6 元買進價值 1 元的股票，預留犯錯空間。</span></div></li>
                                <li class="flex items-start gap-3"><div class="bg-yellow-500 text-slate-900 rounded-full w-6 h-6 flex items-center justify-center font-bold flex-shrink-0">3</div><div><strong class="block text-white">能力圈 (Circle of Competence)</strong><span class="text-sm text-slate-300">只投資自己真正看得懂的生意。</span></div></li>
                            </ul>
                        </div>
                    </div>
                    <div class="border-t border-slate-200 pt-8">
                        <h3 class="text-2xl font-bold text-slate-800 mb-6 flex items-center gap-2"><i data-lucide="briefcase" class="text-yellow-600"></i> 巴菲特經典戰役解析</h3>
                        <div class="grid md:grid-cols-2 gap-6">
                            <div class="bg-red-50 rounded-xl p-6 border border-red-100 hover:shadow-md transition-all"><div class="flex justify-between items-start mb-4"><h4 class="text-xl font-bold text-red-800">1. 可口可樂 (Coca-Cola)</h4><span class="bg-red-200 text-red-800 text-xs px-2 py-1 rounded-full font-bold">1988年買入</span></div><p class="text-slate-700 text-sm mb-3"><b>護城河分析：</b>無可取代的品牌心智佔有率。巴菲特發現，即使稍微漲價，消費者也不會改喝其他品牌（定價權）。</p><div class="flex items-center gap-2 text-xs text-slate-500 bg-white p-2 rounded-lg"><i data-lucide="check-circle" size="14" class="text-green-500"></i><span>持有至今 30+ 年，股息已超過當初投入本金。</span></div></div>
                            <div class="bg-slate-100 rounded-xl p-6 border border-slate-200 hover:shadow-md transition-all"><div class="flex justify-between items-start mb-4"><h4 class="text-xl font-bold text-slate-800">2. 蘋果 (Apple)</h4><span class="bg-slate-300 text-slate-800 text-xs px-2 py-1 rounded-full font-bold">2016年買入</span></div><p class="text-slate-700 text-sm mb-3"><b>護城河分析：</b>強大的生態系黏著度。巴菲特將其視為「消費品」而非單純的科技股，因為用戶一旦進入蘋果生態就很難離開。</p><div class="flex items-center gap-2 text-xs text-slate-500 bg-white p-2 rounded-lg"><i data-lucide="check-circle" size="14" class="text-green-500"></i><span>成為波克夏最大持股，獲利翻倍。</span></div></div>
                        </div>
                    </div>
                </div>
                <div id="tab-taiwan" class="tab-content hidden animate-fade-in"><div class="text-center mb-10"><h3 class="text-3xl font-bold text-slate-800 mb-3">台灣存股名家智慧牆</h3><p class="text-slate-500 max-w-2xl mx-auto">將國際心法應用於台股市場（高殖利率、配息頻繁）。<br/>以下四位名家歸納出最適合台灣人的「存股心法」。</p></div><div id="gurus-grid" class="grid md:grid-cols-2 gap-6"></div></div>
                <div id="tab-golden" class="tab-content hidden animate-fade-in space-y-12">
                    <div class="text-center"><h3 class="text-3xl font-bold text-slate-800 mb-2">黃金投資法則：參數解密</h3><p class="text-slate-500">為什麼這些法則有效？讓我們拆解複利公式背後的數學邏輯。</p></div>
                    <div class="grid md:grid-cols-3 gap-8">
                        <div class="bg-white rounded-2xl p-6 shadow-sm border border-slate-100 relative mt-4"><div class="absolute -top-4 left-6 bg-slate-800 text-white p-3 rounded-xl shadow-md"><i data-lucide="clock" class="text-blue-500"></i></div><div class="mt-8"><h4 class="text-lg font-bold text-slate-800 mb-1">時間 (Time)</h4><p class="text-sm text-slate-500 leading-relaxed bg-slate-50 p-3 rounded-lg">複利效應在後期會呈指數級爆發。投資 30 年的資產翻倍速度遠超 10 年。</p></div></div>
                        <div class="bg-white rounded-2xl p-6 shadow-sm border border-slate-100 relative mt-4"><div class="absolute -top-4 left-6 bg-slate-800 text-white p-3 rounded-xl shadow-md"><i data-lucide="trending-up" class="text-red-500"></i></div><div class="mt-8"><h4 class="text-lg font-bold text-slate-800 mb-1">報酬率 (Rate)</h4><p class="text-sm text-slate-500 leading-relaxed bg-slate-50 p-3 rounded-lg">台股 ETF 長期平均約 5%~8%。黃金法則強調「穩健」，無需冒險賭博。</p></div></div>
                        <div class="bg-white rounded-2xl p-6 shadow-sm border border-slate-100 relative mt-4"><div class="absolute -top-4 left-6 bg-slate-800 text-white p-3 rounded-xl shadow-md"><i data-lucide="target" class="text-green-500"></i></div><div class="mt-8"><h4 class="text-lg font-bold text-slate-800 mb-1">定期投入 (PMT)</h4><p class="text-sm text-slate-500 leading-relaxed bg-slate-50 p-3 rounded-lg">透過每月固定金額投入（定期定額），解決「買在高點」的恐懼，平均成本。</p></div></div>
                    </div>
                    <div class="mt-8 bg-red-50 border-2 border-red-100 rounded-2xl p-8 flex flex-col md:flex-row items-center gap-6 shadow-sm">
                        <div class="bg-red-100 p-4 rounded-full"><i data-lucide="monitor" class="text-red-600" width="40" height="40"></i></div>
                        <div class="flex-1"><h4 class="text-2xl font-bold text-red-700 mb-2">實戰應用：黃金投資法則選股工具</h4><p class="text-slate-600">理論學會了，接下來就是行動！我們為您準備了專屬的選股工具。</p></div>
                        <button onclick="enterScreener()" class="bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-8 rounded-full shadow-lg transition-all flex items-center gap-2 whitespace-nowrap">開啟選股工具 <i data-lucide="external-link" size="18"></i></button>
                    </div>
                </div>
            </div>
        </div>

        <section id="calculator-section" class="bg-slate-100 py-20 px-4">
            <div class="max-w-5xl mx-auto">
                <div class="text-center mb-10"><h2 class="text-3xl md:text-4xl font-bold text-slate-800 mb-4 flex items-center justify-center gap-3"><i data-lucide="calculator" class="text-yellow-600"></i> 雪球複利計算機</h2><p class="text-slate-600">調整下方參數，親眼見證黃金法則的威力。</p></div>
                <div class="bg-white rounded-3xl shadow-xl overflow-hidden flex flex-col md:flex-row border border-slate-200">
                    <div class="p-8 md:p-10 md:w-1/2 bg-white">
                        <h3 class="text-xl font-bold text-slate-700 mb-8 pb-4 border-b border-slate-100">參數設定</h3>
                        <div class="space-y-8">
                            <div class="relative"><label class="block text-sm font-bold text-slate-700 mb-2 flex justify-between">初始資金 (P)<span id="val-initial" class="text-yellow-600">$100,000</span></label><input type="range" id="in-initial" min="0" max="1000000" step="10000" value="100000" class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-slate-800"></div>
                            <div class="relative"><label class="block text-sm font-bold text-slate-700 mb-2 flex justify-between">每月定期投入 (PMT)<span id="val-monthly" class="text-yellow-600">$5,000</span></label><input type="range" id="in-monthly" min="0" max="50000" step="1000" value="5000" class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-slate-800"></div>
                            <div class="relative"><label class="block text-sm font-bold text-slate-700 mb-2 flex justify-between">預期年化報酬率 (R)<span id="val-rate" class="text-yellow-600">6%</span></label><input type="range" id="in-rate" min="1" max="15" step="0.5" value="6" class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-yellow-500"></div>
                            <div class="relative"><label class="block text-sm font-bold text-slate-700 mb-2 flex justify-between">投資年限 (N)<span id="val-years" class="text-yellow-600">20 年</span></label><input type="range" id="in-years" min="5" max="50" value="20" class="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"></div>
                        </div>
                    </div>
                    <div class="p-8 md:p-10 md:w-1/2 bg-slate-900 text-white flex flex-col justify-center relative">
                        <div class="absolute top-0 right-0 p-8 opacity-5"><i data-lucide="bar-chart-3" width="200" height="200"></i></div>
                        <div class="relative z-10 text-center md:text-left">
                            <span class="inline-block bg-slate-800 text-yellow-400 text-xs font-bold px-3 py-1 rounded-full mb-4 border border-slate-700">複利成果預測</span>
                            <p class="text-slate-400 text-sm font-medium mb-1"><span id="res-years">20</span> 年後的總資產</p>
                            <div id="res-total" class="text-4xl md:text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-yellow-300 to-yellow-500 mb-6 break-words tracking-tight">$0</div>
                            <div class="bg-slate-800/50 p-6 rounded-2xl border border-slate-700 backdrop-blur-sm space-y-4">
                                <div class="flex justify-between items-center text-sm border-b border-slate-700 pb-3"><span class="text-slate-400">總投入本金</span><span id="res-principal" class="font-mono text-white">$0</span></div>
                                <div class="flex justify-between items-center text-sm pb-1"><span class="text-green-400 flex items-center gap-1"><i data-lucide="trending-up" size="14"></i> 複利創造財富</span><span id="res-interest" class="font-mono text-green-400 font-bold text-lg">+$0</span></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        
        <footer class="bg-slate-900 text-slate-500 py-10 border-t border-slate-800">
            <div class="max-w-4xl mx-auto px-4 text-center space-y-4">
                <h2 class="text-white font-bold text-lg">黃金投資法則教學網</h2>
                <p class="text-sm max-w-lg mx-auto">免責聲明：本網頁所有數據與計算結果僅供模擬教學使用，不代表未來實際獲利。</p>
            </div>
        </footer>
    </div>

    <div id="screener-view" class="flex-1 flex flex-col hidden h-screen" x-data="app()">
        <header class="bg-white px-4 py-3 border-b border-slate-200 flex justify-between items-center sticky top-0 z-50 shadow-sm shrink-0">
            <div class="flex items-center gap-4">
                <button onclick="exitScreener()" class="text-slate-500 hover:text-slate-800 transition-colors flex items-center gap-1 text-sm font-bold">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
                    回教學頁
                </button>
                <div class="font-bold text-lg text-blue-700 tracking-tight flex items-center gap-1">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                    Pocket<span class="text-slate-900">Screener</span>
                </div>
            </div>
            <div class="flex flex-col items-end">
                <div class="text-[10px] text-slate-400">更新: {update_time}</div>
                <div class="text-[10px] font-mono text-white bg-purple-600 px-1.5 rounded">V2.3</div>
            </div>
        </header>

        <main class="flex-1 overflow-y-auto no-scrollbar pb-32">
            <div class="m-3 bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden" :class="showFilter ? '' : 'h-14'">
                <div class="p-4 bg-slate-50 border-b border-slate-100 flex justify-between items-center cursor-pointer" @click="showFilter = !showFilter">
                    <h2 class="text-sm font-bold text-slate-600 uppercase flex items-center gap-2">篩選條件</h2>
                    <div class="flex gap-3"><span x-show="!showFilter && filters.length > 0" class="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full" x-text="filters.length + ' 個條件'"></span><svg class="w-4 h-4 text-slate-400 transform transition-transform" :class="showFilter ? 'rotate-180' : ''" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg></div>
                </div>
                <div x-show="showFilter" class="p-4 pt-2">
                    <button @click="applyDepositStrategy()" class="w-full mb-4 bg-gradient-to-r from-yellow-400 to-yellow-600 hover:from-yellow-500 hover:to-yellow-700 text-white py-3 rounded-lg font-bold text-md shadow-lg flex items-center justify-center gap-2 transition-all transform active:scale-95">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v13m0-13V6a2 2 0 112 2h-2zm0 0V5.5A2.5 2.5 0 109.5 8H12zm-7 4h14M5 12a2 2 0 110-4h14a2 2 0 110 4M5 12v7a2 2 0 002 2h10a2 2 0 002-2v-7"></path></svg>
                        一鍵套用「黃金存股 8 法則」
                    </button>
                    <div class="flex flex-wrap gap-2 mb-4 min-h-[30px]"><template x-for="(filter, index) in filters" :key="index"><div class="flex items-center gap-1 bg-blue-50 text-blue-700 px-3 py-1.5 rounded-full border border-blue-100 text-sm shadow-sm"><span class="font-medium" x-text="getLabel(filter)"></span><button @click="removeFilter(index)" class="ml-1 text-blue-400 hover:text-blue-800 font-bold">×</button></div></template></div>
                    <div class="flex flex-col gap-3 bg-slate-50 p-3 rounded-lg border border-slate-200">
                        <select x-model="newFilter.type" class="w-full p-2.5 rounded-lg border border-slate-300 text-sm font-medium bg-white outline-none focus:ring-2 focus:ring-blue-500">
                            <option value="" disabled selected>選擇指標...</option>
                            <optgroup label="💰 黃金存股 8 法則">
                                <option value="eps_ttm">近一年 EPS (元)</option>
                                <option value="eps_avg">5年平均 EPS (元)</option>
                                <option value="yield_avg">5年平均殖利率 (%)</option>
                                <option value="cons_div">連續配發股利 (年)</option>
                                <option value="roe_avg">5年平均 ROE (%)</option>
                                <option value="core_purity">本業純度 (%)</option>
                                <option value="gm_stability">毛利變動度 (%)</option>
                                <option value="payout_ratio">盈餘發放率 (%)</option>
                            </optgroup>
                            <optgroup label="📊 其他指標">
                                <option value="yield">現金殖利率 (%)</option>
                                <option value="pe">本益比 P/E</option>
                                <option value="pb">股價淨值比 P/B</option>
                                <option value="rev_growth">營收成長 YoY (%)</option>
                                <option value="gross_margin">毛利率 (%)</option>
                                <option value="ma_bull">站上月線 (是/否)</option>
                            </optgroup>
                        </select>
                        <div class="flex gap-2" x-show="newFilter.type !== 'ma_bull'"><select x-model="newFilter.operator" class="w-1/3 p-2.5 rounded-lg border border-slate-300 text-sm bg-white"><option value=">=">大於</option><option value="<=">小於</option></select><input type="number" x-model="newFilter.value" class="w-2/3 p-2.5 rounded-lg border border-slate-300 text-sm" placeholder="數值"></div>
                        <button @click="addFilter()" class="w-full bg-blue-600 hover:bg-blue-700 text-white py-2.5 rounded-lg font-bold text-sm shadow-md transition-colors">加入篩選</button>
                    </div>
                </div>
            </div>
            <div class="px-4 py-2 flex justify-between items-center border-b border-slate-200 mx-2 pb-2 bg-slate-100">
                <div class="text-sm font-medium text-slate-500">符合: <span x-text="filteredStocks.length"></span> 檔<span x-show="displayCount < filteredStocks.length" class="text-xs text-slate-400 ml-1">(前 <span x-text="displayCount"></span>)</span></div>
                <div class="flex items-center gap-2">
                    <div class="text-xs text-slate-400">排序:</div>
                    <select x-model="sortKey" class="p-1 rounded border border-slate-300 text-sm font-bold text-slate-700 bg-white outline-none focus:ring-2 focus:ring-blue-500">
                        <option value="yield_avg">5年平均殖利率</option>
                        <option value="roe_avg">5年平均 ROE</option>
                        <option value="eps_avg">5年平均 EPS</option>
                        <option value="core_purity">本業純度</option>
                        <option value="cons_div">配息年數</option>
                        <option value="yield">目前殖利率</option>
                        <option value="id">股票代號</option>
                    </select>
                    <button @click="sortDesc = !sortDesc" class="p-1.5 bg-white rounded-md border border-slate-200 shadow-sm text-slate-600 active:bg-slate-100"><span x-show="sortDesc">⬇️</span><span x-show="!sortDesc">⬆️</span></button>
                </div>
            </div>
            <div class="px-3 py-3 space-y-3">
                <template x-for="stock in filteredStocks.slice(0, displayCount)" :key="stock.id">
                    <div class="bg-white p-4 rounded-xl border border-slate-100 shadow-[0_4px_20px_-4px_rgba(0,0,0,0.05)] transition-all hover:shadow-md">
                        <div class="flex gap-1 mb-2 overflow-x-auto no-scrollbar"><template x-for="tag in stock.tags"><span class="text-[10px] font-bold px-2 py-0.5 rounded-md whitespace-nowrap" :class="tag.includes('黃金') ? 'bg-gradient-to-r from-yellow-400 to-yellow-600 text-white shadow-sm' : (tag.includes('高') ? 'bg-rose-100 text-rose-700' : 'bg-blue-100 text-blue-700')" x-text="tag"></span></template></div>
                        <div class="flex justify-between items-center mb-3">
                            <div class="w-1/3">
                                <a :href="`https://tw.stock.yahoo.com/quote/${{stock.id}}`" target="_blank" class="flex items-center gap-2 hover:text-blue-600 transition-colors">
                                    <span class="text-2xl font-bold text-slate-900 hover:text-blue-600 cursor-pointer" x-text="stock.id"></span>
                                    <svg class="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path></svg>
                                </a>
                                <div class="text-sm text-slate-600 font-medium truncate" x-text="stock.name"></div>
                                <div class="text-[10px] text-slate-400 mt-1 flex flex-col"><span :class="sortKey==='pe'?'text-blue-600 font-bold':''">P/E: <span x-text="stock.pe>0?stock.pe:'-'"></span></span><span>P/B: <span x-text="stock.pb>0?stock.pb:'-'"></span></span></div>
                            </div>
                            <div class="flex-1 h-10 px-2 flex items-center justify-center"><template x-if="stock.sparkline.length > 2"><svg class="w-full h-full overflow-visible" viewBox="0 0 100 30" preserveAspectRatio="none"><path :d="getSparklinePath(stock.sparkline)" fill="none" stroke-width="2" :stroke="stock.sparkline[stock.sparkline.length-1] >= stock.sparkline[0] ? '#ef4444' : '#10b981'" stroke-linecap="round" stroke-linejoin="round" /></svg></template></div>
                            <div class="w-1/3 text-right"><div class="text-xl font-bold text-slate-800" x-text="stock.price"></div><div class="text-xs font-bold" :class="stock.rev_growth>0?'text-red-500':'text-green-500'">YoY: <span x-text="stock.rev_growth!=0?stock.rev_growth+'%':'-'"></span></div><div class="text-[10px] mt-1 text-slate-400">殖: <span class="font-bold text-emerald-600" x-text="stock.yield>0?stock.yield+'%':'-'"></span></div></div>
                        </div>
                        <div class="grid grid-cols-4 gap-1 bg-slate-50 p-2 rounded-lg border border-slate-100 text-center">
                            <div :class="sortKey==='roe_avg'?'bg-blue-50 ring-1 ring-blue-200 rounded':''"><div class="text-[10px] text-slate-400">5年ROE</div><div class="font-bold text-sm text-blue-600" x-text="stock.roe_avg!=0?stock.roe_avg+'%':'-'"></div></div>
                            <div :class="sortKey==='yield_avg'?'bg-emerald-50 ring-1 ring-emerald-200 rounded':''"><div class="text-[10px] text-slate-400">5年殖利</div><div class="font-bold text-sm text-emerald-600" x-text="stock.yield_avg>0?stock.yield_avg+'%':'-'"></div></div>
                            <div :class="sortKey==='core_purity'?'bg-purple-50 ring-1 ring-purple-200 rounded':''"><div class="text-[10px] text-slate-400">本業純度</div><div class="font-bold text-sm text-purple-600" x-text="stock.core_purity!=0?stock.core_purity+'%':'-'"></div></div>
                            <div :class="sortKey==='cons_div'?'bg-amber-50 ring-1 ring-amber-200 rounded':''"><div class="text-[10px] text-slate-400">配息年</div><div class="font-bold text-sm text-amber-600" x-text="stock.cons_div"></div></div>
                        </div>
                    </div>
                </template>
                <div x-show="displayCount < filteredStocks.length" class="text-center py-4"><button @click="displayCount+=20" class="bg-white border border-slate-300 text-slate-600 px-6 py-2 rounded-full text-sm font-bold shadow-sm hover:bg-slate-50 transition-all">載入更多... (<span x-text="filteredStocks.length-displayCount"></span>)</button></div>
            </div>
        </main>
    </div>

    <script>
        // --- VIEW 1 LOGIC (Welcome Page) ---
        const GURUS = [
            {{ id: 'shi', name: '施昇輝 (樂活大叔)', title: '暢銷理財作家', icon: '🧘', quote: '投資是為了讓生活更美好，而不是讓你睡不著覺。', philosophy: '推崇「0050/0056」簡單投資法。認為普通人不必鑽研財報，只要跟隨大盤指數（0050）或高股息（0056），就能取得超越定存的報酬。', highlight: 'K<20買，K>80賣 (針對0050的操作口訣)', searchQuery: '施昇輝 0050 操作心法', articleTitle: '【樂活投資】為什麼我只買0050？', articleContent: ['施昇輝認為，人生有許多比投資更重要的事情。選股非常耗神，且容易看錯。', '核心策略一：只買 0050（台灣50）。因為它包含了台灣市值最大的50家公司，大到不能倒，且每年穩定配息。', '核心策略二：日K值投資法。當日K值小於20時，代表市場過度恐慌，是大膽買進的時機；當日K值大於80時，代表市場過熱，可以分批賣出獲利了結。', '結論：透過簡單的紀律，你可以把時間花在陪伴家人與享受生活，而不是盯著盤面。'] }},
            {{ id: 'chen', name: '陳重銘 (不敗教主)', title: '資深投資達人', icon: '🏫', quote: '打造你的「資產」，讓資產幫你買單，而不是用勞力買單。', philosophy: '強調「不敗」就是不賠錢，透過持有績優股或 ETF 領取股息，並將股息「再投入」買股，創造複利滾雪球效應。', highlight: '存股就像種樹，樹長大會生股子股孫', searchQuery: '陳重銘 存股 不敗教主', articleTitle: '【不敗心法】讓股息幫你繳房貸', articleContent: ['陳重銘老師原本是領死薪水的流浪教師，靠著存股滾出數千萬資產。', '核心觀念：即使薪水低，也要擠出錢來買進資產。他強調「不敗」的關鍵在於買進不會倒的公司（如金融股、ETF）。', '股息再投入：拿到股息絕對不能花掉，要立刻買進更多的股票。這樣明年的股息會更多，形成正向循環。', '重點：不要在意股價短期的漲跌，要專注於手中持有的「股數」是否增加。'] }},
            {{ id: 'emily', name: '艾蜜莉 (小資女)', title: '財經作家', icon: '🚦', quote: '好公司要在「便宜價」買進，並預留「安全邊際」。', philosophy: '獨創「紅綠燈估價法」，將股票分為便宜、合理、昂貴三種價格。強調在利空時勇敢買進績優股，耐心等待價格回歸。', highlight: '逆勢價值投資，人棄我取', searchQuery: '艾蜜莉 定存股 紅綠燈', articleTitle: '【小資翻身】紅綠燈估價法教學', articleContent: ['艾蜜莉將價值投資量化為簡單的紅綠燈號。', '綠燈（便宜價）：當好公司遇到倒楣事（如食安風暴、短期匯損），股價跌到便宜價以下，就是全力買進的時機。', '黃燈（合理價）：持有並領取股息，或分批調節。', '紅燈（昂貴價）：分批賣出，保留現金等待下一次機會。', '這套方法非常適合資金不多、想要穩健獲利的小資族。'] }},
            {{ id: 'warren', name: '周文偉 (華倫老師)', title: '流浪教師變千萬富翁', icon: '🛍️', quote: '時間是好公司的朋友，卻是壞公司的敵人。', philosophy: '專注於「民生消費股」（如食品、電信、環保），因為這些產業受景氣影響小，具備護城河與重複消費特性，適合長期持有。', highlight: '讓每一塊錢都替你賺錢', searchQuery: '華倫老師 存股 養對股票賺千萬', articleTitle: '【生活選股】從逛超市挖掘定存股', articleContent: ['華倫老師喜歡從生活中找股票，例如大家每天都要用的豆腐、沙拉油、電信服務、廢棄物處理。', '這類公司的特色是：產品具有重複消費性、市場獨佔或寡佔、不需要一直更新昂貴的設備。', '策略：只要公司獲利穩定成長，就長期持有，只買不賣。利用時間的複利，讓資產像滾雪球一樣越滾越大。'] }}
        ];

        lucide.createIcons();

        function switchTab(tabName) {{
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            document.getElementById('tab-' + tabName).classList.remove('hidden');
            document.querySelectorAll('.tab-btn').forEach(btn => {{ btn.className = 'tab-btn px-6 py-3 rounded-full font-bold transition-all flex items-center gap-2 bg-white text-slate-600 hover:bg-slate-100'; }});
            document.getElementById('btn-' + tabName).className = 'tab-btn px-6 py-3 rounded-full font-bold transition-all flex items-center gap-2 bg-slate-800 text-yellow-400 shadow-lg scale-105';
        }}
        switchTab('buffett');

        const guruGrid = document.getElementById('gurus-grid');
        GURUS.forEach(guru => {{
            const card = document.createElement('div');
            card.className = "bg-slate-50 rounded-2xl p-6 border border-slate-200 hover:shadow-lg transition-all hover:border-yellow-400 group relative overflow-hidden flex flex-col";
            card.innerHTML = `<div class="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity"><span class="text-6xl">${{guru.icon}}</span></div><div class="flex items-start gap-4 mb-4"><div class="w-16 h-16 bg-white rounded-full flex items-center justify-center text-3xl shadow-sm border border-slate-100 flex-shrink-0">${{guru.icon}}</div><div><h4 class="text-xl font-bold text-slate-800">${{guru.name}}</h4><span class="text-xs font-semibold bg-slate-200 text-slate-600 px-2 py-1 rounded-full">${{guru.title}}</span></div></div><div class="mb-4 flex-grow"><p class="text-slate-700 text-sm leading-relaxed mb-3">${{guru.philosophy}}</p><div class="bg-yellow-50 p-3 rounded-lg border border-yellow-100"><p class="text-xs text-yellow-800 font-bold flex items-center gap-2"><i data-lucide="lightbulb" size="12"></i> 核心心法：${{guru.highlight}}</p></div></div><div class="border-t border-slate-200 pt-4 mt-auto"><button onclick="openModal('${{guru.id}}')" class="w-full bg-slate-800 hover:bg-slate-700 text-white text-sm font-bold py-2.5 rounded-lg transition-colors flex items-center justify-center gap-2"><i data-lucide="book-open" size="16"></i> 閱讀投資策略</button></div>`;
            guruGrid.appendChild(card);
        }});
        lucide.createIcons();

        const modalOverlay = document.getElementById('modal-overlay');
        const modalContainer = document.getElementById('modal-container');
        let currentSearchQuery = '';

        function openModal(guruId) {{
            const guru = GURUS.find(g => g.id === guruId);
            if (!guru) return;
            document.getElementById('modal-icon').textContent = guru.icon;
            document.getElementById('modal-title').textContent = guru.articleTitle;
            document.getElementById('modal-subtitle').textContent = `專家：${{guru.name}}`;
            const bodyDiv = document.getElementById('modal-body');
            bodyDiv.innerHTML = '';
            guru.articleContent.forEach(p => {{
                const pTag = document.createElement('p'); pTag.className = "border-l-2 border-slate-200 pl-3"; pTag.textContent = p; bodyDiv.appendChild(pTag);
            }});
            document.getElementById('modal-highlight').textContent = guru.highlight;
            currentSearchQuery = guru.searchQuery;
            modalOverlay.classList.remove('hidden');
            setTimeout(() => {{ modalOverlay.classList.remove('opacity-0'); modalContainer.classList.remove('scale-95'); modalContainer.classList.add('scale-100'); }}, 10);
        }}

        function closeModal() {{
            modalOverlay.classList.add('opacity-0'); modalContainer.classList.remove('scale-100'); modalContainer.classList.add('scale-95');
            setTimeout(() => {{ modalOverlay.classList.add('hidden'); }}, 300);
        }}

        document.getElementById('modal-search-btn').onclick = function() {{ window.open(`https://www.google.com/search?q=${{encodeURIComponent(currentSearchQuery)}}`, '_blank'); }};

        const inputs = {{ initial: document.getElementById('in-initial'), monthly: document.getElementById('in-monthly'), rate: document.getElementById('in-rate'), years: document.getElementById('in-years') }};
        const displays = {{ initial: document.getElementById('val-initial'), monthly: document.getElementById('val-monthly'), rate: document.getElementById('val-rate'), years: document.getElementById('val-years'), total: document.getElementById('res-total'), principal: document.getElementById('res-principal'), interest: document.getElementById('res-interest'), resYears: document.getElementById('res-years') }};

        function formatCurrency(num) {{ return new Intl.NumberFormat('zh-TW', {{ style: 'currency', currency: 'TWD', maximumFractionDigits: 0 }}).format(num); }}
        function calculate() {{
            const p = Number(inputs.initial.value); const pmt = Number(inputs.monthly.value); const r = Number(inputs.rate.value); const n = Number(inputs.years.value);
            displays.initial.textContent = formatCurrency(p); displays.monthly.textContent = formatCurrency(pmt); displays.rate.textContent = r + '%'; displays.years.textContent = n + ' 年'; displays.resYears.textContent = n;
            let total = p; for (let i = 0; i < n * 12; i++) {{ total = total * (1 + r / 100 / 12) + pmt; }}
            const totalInvested = p + (pmt * 12 * n); const interestEarned = total - totalInvested;
            displays.total.textContent = formatCurrency(total); displays.principal.textContent = formatCurrency(totalInvested); displays.interest.textContent = '+' + formatCurrency(interestEarned);
        }}
        Object.values(inputs).forEach(input => {{ input.addEventListener('input', calculate); }});
        calculate();

        function scrollToSection(id) {{ document.getElementById(id).scrollIntoView({{ behavior: 'smooth' }}); }}

        // --- NAVIGATION LOGIC ---
        function enterScreener() {{
            document.getElementById('welcome-view').classList.add('hidden');
            document.getElementById('screener-view').classList.remove('hidden');
            window.scrollTo(0,0);
        }}
        function exitScreener() {{
            document.getElementById('screener-view').classList.add('hidden');
            document.getElementById('welcome-view').classList.remove('hidden');
            window.scrollTo(0,0);
        }}

        // --- VIEW 2 LOGIC (Screener) ---
        function app() {{
            return {{
                stocks: {json_db}, filters: [], newFilter: {{ type: 'roe_avg', operator: '>=', value: 15 }}, showFilter: true, sortKey: 'yield_avg', sortDesc: true, displayCount: 20,
                
                applyDepositStrategy() {{
                    this.filters = [
                        {{ type: 'eps_ttm', operator: '>=', value: 1 }},   
                        {{ type: 'eps_avg', operator: '>=', value: 2 }},   
                        {{ type: 'yield_avg', operator: '>=', value: 5 }}, 
                        {{ type: 'cons_div', operator: '>=', value: 10 }}, 
                        {{ type: 'roe_avg', operator: '>=', value: 15 }},
                        {{ type: 'core_purity', operator: '>=', value: 80 }},
                        {{ type: 'gm_stability', operator: '<=', value: 5 }},
                        {{ type: 'payout_ratio', operator: '>=', value: 60 }},
                        {{ type: 'payout_ratio', operator: '<=', value: 100 }}
                    ];
                    this.sortKey = 'yield_avg';
                    this.displayCount = 20;
                    alert('✅ 已套用「黃金存股 8 法則」！(含純度/穩定度/發放率)');
                }},

                get filteredStocks() {{
                    let res = this.stocks;
                    if (this.filters.length > 0) {{
                        res = res.filter(s => this.filters.every(f => {{
                            let v = s[f.type];
                            if (f.type === 'ma_bull') return v === true;
                            return f.operator === '>=' ? v >= parseFloat(f.value) : v <= parseFloat(f.value);
                        }}));
                    }}
                    return res.sort((a, b) => (this.sortDesc ? (b[this.sortKey] || -999) - (a[this.sortKey] || -999) : (a[this.sortKey] || -999) - (b[this.sortKey] || -999)));
                }},
                getLabel(f) {{ const map = {{ 'roe_avg': '5年ROE', 'eps_ttm': 'EPS', 'eps_avg': '5年EPS', 'gross_margin': '毛利率', 'yield': '殖利率', 'yield_avg': '5年殖利', 'pe': 'PE', 'pb': 'PB', 'rev_growth': '營收YoY', 'vol': '成交量', 'ma_bull': '站上月線', 'cons_div': '連續配息', 'core_purity': '本業純度', 'gm_stability': '毛利變動', 'payout_ratio': '發放率' }}; return f.type === 'ma_bull' ? map[f.type] : `${{map[f.type]}} ${{f.operator}} ${{f.value}}`; }},
                addFilter() {{ if (this.newFilter.type) this.filters.push(this.newFilter.type === 'ma_bull' ? {{ type: 'ma_bull', operator: '=', value: 0 }} : {{ ...this.newFilter }}); this.displayCount = 20; }},
                removeFilter(i) {{ this.filters.splice(i, 1); }},
                getSparklinePath(d) {{ if (!d.length) return ""; const w=100, h=30, min=Math.min(...d), max=Math.max(...d), r=max-min||1, sx=w/(d.length-1); return d.map((p,i)=>`${{i==0?'M':'L'}} ${{i*sx}} ${{h-((p-min)/r)*h}}`).join(' '); }},
                init() {{ this.$watch('filters', ()=>this.displayCount=20); this.$watch('sortKey', ()=>this.displayCount=20); }}
            }}
        }}
    </script>
</body>
</html>"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)
