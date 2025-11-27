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
from datetime import datetime
from fake_useragent import UserAgent
import yfinance as yf

# 1. 忽略警告與錯誤雜訊
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
print(f"📥 [1/4] 正在獲取全台股清單 ({datetime.now().strftime('%H:%M:%S')})...")

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
# 2. 批次下載股價 (Batch Download)
# ==========================================
print("\n📥 [2/4] 啟動批次股價下載 (Chunk Size: 100)...")

processed_data = {}
BATCH_SIZE = 100
chunks = [all_stocks[i:i + BATCH_SIZE] for i in range(0, len(all_stocks), BATCH_SIZE)]
total_batches = len(chunks)

for i, chunk in enumerate(chunks):
    tickers = [s['ticker'] for s in chunk]
    # 強制刷新進度
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
                
                # 簡易過濾
                if vol < 5: continue

                price = round(close[-1], 2)
                ma20 = sum(close[-20:]) / 20 if len(close) >= 20 else 0
                
                processed_data[t] = {
                    "id": stock['id'], "name": stock['name'],
                    "price": price, "vol": vol,
                    "sparkline": [round(x, 2) for x in close], 
                    "ma_bull": price > ma20,
                    # 初始化欄位
                    "eps_ttm": 0, "eps_avg": 0, 
                    "roe_ttm": 0, "roe_avg": 0, "roa": 0,
                    "gross_margin": 0, "op_margin": 0, 
                    "pe": 0, "pb": 0, "yield": 0, "yield_avg": 0,
                    "rev_growth": 0, "net_growth": 0, "cons_div": 0,
                    "tags": [] 
                }
            except: continue
    except: pass

print(f"\n✅ 股價獲取完成！有效: {len(processed_data)} 檔")

# ==========================================
# 3. 補充完整財報資料 (含配息與殖利率)
# ==========================================
print("\n📥 [3/4] 正在抓取詳細財報指標 (抗封鎖模式)...")
print("   ⚠️ 新增：計算連續配息年份與5年平均殖利率。")

def fetch_detailed_stats(ticker):
    time.sleep(random.uniform(0.5, 2.0)) # 隨機延遲
    try:
        stock = yf.Ticker(ticker)
        # 嘗試獲取 info
        try:
            info = stock.info
        except:
            time.sleep(2)
            stock = yf.Ticker(ticker)
            info = stock.info

        # 基本面
        pe = round(info.get('trailingPE', 0), 2)
        pb = round(info.get('priceToBook', 0), 2)
        eps_ttm = info.get('trailingEps', 0)
        roe_ttm = round(info.get('returnOnEquity', 0) * 100, 2)
        roa = round(info.get('returnOnAssets', 0) * 100, 2)
        gross_margin = round(info.get('grossMargins', 0) * 100, 2)
        op_margin = round(info.get('operatingMargins', 0) * 100, 2)
        rev_growth = round(info.get('revenueGrowth', 0) * 100, 2)
        
        # 殖利率
        div_yield = 0
        if info.get('dividendRate') and info.get('regularMarketPrice'):
             div_yield = round((info['dividendRate'] / info['regularMarketPrice']) * 100, 2)
        
        # [新增] 5年平均殖利率 (直接從 info 拿)
        yield_avg = info.get('fiveYearAvgDividendYield', 0)
        if yield_avg is None: yield_avg = 0
        else: yield_avg = round(yield_avg, 2)

        # [新增] 連續配息計算
        cons_div = 0
        try:
            # 抓取 15 年配息紀錄來計算連續配息
            divs = stock.history(period="15y")['Dividends']
            if not divs.empty:
                # 依年份加總
                yearly_divs = divs.groupby(divs.index.year).sum()
                current_y = datetime.now().year
                # 從去年開始往前推 (今年可能還沒配)
                check_year = current_y - 1
                # 如果去年沒配，檢查前年 (容許一年空窗期，例如剛好資料還沒更新)
                if check_year not in yearly_divs.index or yearly_divs.loc[check_year] == 0:
                    if (check_year - 1) in yearly_divs.index and yearly_divs.loc[check_year - 1] > 0:
                        check_year -= 1
                
                # 開始回推
                while check_year in yearly_divs.index and yearly_divs.loc[check_year] > 0:
                    cons_div += 1
                    check_year -= 1
        except:
            # 若抓取失敗，至少看 info 有沒有殖利率，有的話算 1 年
            if div_yield > 0: cons_div = 1

        return {
            "pe": pe, "pb": pb, "yield": div_yield, "yield_avg": yield_avg,
            "eps_ttm": eps_ttm, "eps_avg": eps_ttm, # 暫用 TTM
            "roe_ttm": roe_ttm, "roe_avg": roe_ttm, # 暫用 TTM
            "roa": roa, "gross_margin": gross_margin, "op_margin": op_margin,
            "rev_growth": rev_growth, "cons_div": cons_div
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
    future_to_ticker = {executor.submit(fetch_detailed_stats, t): t for t in tickers_to_enrich}
    
    for future in concurrent.futures.as_completed(future_to_ticker):
        t = future_to_ticker[future]
        count += 1
        
        try:
            stats = future.result()
            if stats:
                processed_data[t].update(stats)
                
                tags = []
                if processed_data[t]['yield'] > 5: tags.append("💰高殖利")
                if processed_data[t]['cons_div'] >= 10: tags.append("🏆連續配息") # 新增標籤
                if processed_data[t]['roe_ttm'] > 15: tags.append("🔥高ROE")
                if processed_data[t]['rev_growth'] > 20: tags.append("🚀高成長")
                if processed_data[t]['ma_bull']: tags.append("📈站上月線")
                processed_data[t]['tags'] = tags
                
                enriched_count += 1
        except: pass

        # 進度刷新
        pct = (count / total) * 100
        sys.stdout.write(f"\r   - 進度: {count}/{total} ({pct:.1f}%) | 成功獲取: {enriched_count} 檔")
        sys.stdout.flush()

print(f"\n\n✅ 資料獲取作業結束。")

# 轉 JSON
final_db = list(processed_data.values())
json_db = json.dumps(final_db, cls=NpEncoder, ensure_ascii=False)

# --- 統計報告 ---
print("\n" + "="*35)
print("📊 TW-PocketScreener v1.8 執行報告")
print("="*35)
print(f"📋 監測總數 : {len(all_stocks)} 檔")
print(f"✅ 股價有效 : {len(processed_data)} 檔")
print(f"💎 財報完整 : {enriched_count} 檔")
print("="*35 + "\n")

# ==========================================
# 4. 生成 HTML (v1.8)
# ==========================================
update_time = datetime.now().strftime('%Y-%m-%d %H:%M')

html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>TW-PocketScreener v1.8</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/alpinejs/3.13.3/cdn.min.js" defer></script>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap" rel="stylesheet">
    <style>body {{ font-family: 'Noto Sans TC', sans-serif; -webkit-tap-highlight-color: transparent; }} .no-scrollbar::-webkit-scrollbar {{ display: none; }} [x-cloak] {{ display: none !important; }}</style>
</head>
<body class="bg-slate-100 text-slate-800 h-screen flex flex-col" x-data="app()">
    <header class="bg-white px-4 py-3 border-b border-slate-200 flex justify-between items-center sticky top-0 z-50 shadow-sm">
        <div class="font-bold text-lg text-blue-700 tracking-tight flex items-center gap-1">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
            Pocket<span class="text-slate-900">Screener</span>
        </div>
        <div class="text-[10px] text-slate-400">{update_time}</div>
    </header>
    <main class="flex-1 overflow-y-auto no-scrollbar pb-32">
        <div class="m-3 bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden" :class="showFilter ? '' : 'h-14'">
            <div class="p-4 bg-slate-50 border-b border-slate-100 flex justify-between items-center cursor-pointer" @click="showFilter = !showFilter">
                <h2 class="text-sm font-bold text-slate-600 uppercase flex items-center gap-2">篩選條件</h2>
                <div class="flex gap-3"><span x-show="!showFilter && filters.length > 0" class="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full" x-text="filters.length + ' 個條件'"></span><svg class="w-4 h-4 text-slate-400 transform transition-transform" :class="showFilter ? 'rotate-180' : ''" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg></div>
            </div>
            <div x-show="showFilter" class="p-4 pt-2">
                <div class="flex flex-wrap gap-2 mb-4 min-h-[30px]"><template x-for="(filter, index) in filters" :key="index"><div class="flex items-center gap-1 bg-blue-50 text-blue-700 px-3 py-1.5 rounded-full border border-blue-100 text-sm shadow-sm"><span class="font-medium" x-text="getLabel(filter)"></span><button @click="removeFilter(index)" class="ml-1 text-blue-400 hover:text-blue-800 font-bold">×</button></div></template></div>
                <div class="flex flex-col gap-3 bg-slate-50 p-3 rounded-lg border border-slate-200">
                    <select x-model="newFilter.type" class="w-full p-2.5 rounded-lg border border-slate-300 text-sm font-medium bg-white outline-none focus:ring-2 focus:ring-blue-500">
                        <option value="" disabled selected>選擇指標...</option>
                        <optgroup label="💰 獲利能力">
                            <option value="roe_avg">5年平均 ROE (%)</option>
                            <option value="roe_ttm">近一年 ROE (%)</option>
                            <option value="eps_avg">5年平均 EPS (元)</option>
                            <option value="eps_ttm">近一年 EPS (元)</option>
                            <option value="roa">資產報酬率 ROA (%)</option>
                            <option value="gross_margin">毛利率 (%)</option>
                            <option value="op_margin">營業利益率 (%)</option>
                        </optgroup>
                        <optgroup label="📊 估值與股利">
                            <option value="yield">現金殖利率 (%)</option>
                            <option value="yield_avg">5年平均殖利率 (%)</option>
                            <option value="pe">本益比 P/E</option>
                            <option value="pb">股價淨值比 P/B</option>
                        </optgroup>
                        <optgroup label="🚀 成長與籌碼">
                            <option value="rev_growth">營收年增率 YoY (%)</option>
                            <option value="cons_div">連續配發股利 (年)</option>
                            <option value="vol">成交均量 (張)</option>
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
            <div class="flex items-center gap-2"><div class="text-xs text-slate-400">排序:</div><select x-model="sortKey" class="text-sm font-bold text-slate-700 bg-transparent border-none outline-none focus:ring-0 cursor-pointer text-right dir-rtl"><option value="yield">殖利率</option><option value="yield_avg">5年殖利</option><option value="cons_div">配息年數</option><option value="roe_avg">ROE</option><option value="pe">PE</option><option value="rev_growth">營收成長</option><option value="id">代號</option></select><button @click="sortDesc = !sortDesc" class="p-1.5 bg-white rounded-md border border-slate-200 shadow-sm text-slate-600 active:bg-slate-100"><span x-show="sortDesc">⬇️</span><span x-show="!sortDesc">⬆️</span></button></div>
        </div>
        <div class="px-3 py-3 space-y-3">
            <template x-for="stock in filteredStocks.slice(0, displayCount)" :key="stock.id">
                <div class="bg-white p-4 rounded-xl border border-slate-100 shadow-[0_4px_20px_-4px_rgba(0,0,0,0.05)] transition-all hover:shadow-md">
                    <div class="flex gap-1 mb-2 overflow-x-auto no-scrollbar"><template x-for="tag in stock.tags"><span class="text-[10px] font-bold px-2 py-0.5 rounded-md whitespace-nowrap" :class="tag.includes('配息') ? 'bg-amber-100 text-amber-700' : (tag.includes('高') ? 'bg-rose-100 text-rose-700' : 'bg-blue-100 text-blue-700')" x-text="tag"></span></template></div>
                    <div class="flex justify-between items-center mb-3">
                        <div class="w-1/3"><div class="flex items-center gap-2"><span class="text-2xl font-bold text-slate-900" x-text="stock.id"></span></div><div class="text-sm text-slate-600 font-medium truncate" x-text="stock.name"></div><div class="text-[10px] text-slate-400 mt-1 flex flex-col"><span :class="sortKey==='pe'?'text-blue-600 font-bold':''">P/E: <span x-text="stock.pe>0?stock.pe:'-'"></span></span><span>P/B: <span x-text="stock.pb>0?stock.pb:'-'"></span></span></div></div>
                        <div class="flex-1 h-10 px-2 flex items-center justify-center"><template x-if="stock.sparkline.length > 2"><svg class="w-full h-full overflow-visible" viewBox="0 0 100 30" preserveAspectRatio="none"><path :d="getSparklinePath(stock.sparkline)" fill="none" stroke-width="2" :stroke="stock.sparkline[stock.sparkline.length-1] >= stock.sparkline[0] ? '#ef4444' : '#10b981'" stroke-linecap="round" stroke-linejoin="round" /></svg></template></div>
                        <div class="w-1/3 text-right"><div class="text-xl font-bold text-slate-800" x-text="stock.price"></div><div class="text-xs font-bold" :class="stock.rev_growth>0?'text-red-500':'text-green-500'">YoY: <span x-text="stock.rev_growth!=0?stock.rev_growth+'%':'-'"></span></div><div class="text-[10px] mt-1 text-slate-400">殖: <span class="font-bold text-emerald-600" x-text="stock.yield>0?stock.yield+'%':'-'"></span></div></div>
                    </div>
                    <div class="grid grid-cols-4 gap-1 bg-slate-50 p-2 rounded-lg border border-slate-100 text-center">
                        <div :class="sortKey==='roe_avg'?'bg-blue-50 ring-1 ring-blue-200 rounded':''"><div class="text-[10px] text-slate-400">ROE</div><div class="font-bold text-sm text-blue-600" x-text="stock.roe_avg!=0?stock.roe_avg+'%':'-'"></div></div>
                        <div :class="sortKey==='yield_avg'?'bg-emerald-50 ring-1 ring-emerald-200 rounded':''"><div class="text-[10px] text-slate-400">5年殖利</div><div class="font-bold text-sm text-emerald-600" x-text="stock.yield_avg>0?stock.yield_avg+'%':'-'"></div></div>
                        <div class="border-l border-slate-200"><div class="text-[10px] text-slate-400">EPS</div><div class="font-bold text-sm text-purple-600" x-text="stock.eps_ttm!=0?stock.eps_ttm:'-'"></div></div>
                        <div class="border-l border-slate-200" :class="sortKey==='cons_div'?'bg-amber-50 ring-1 ring-amber-200 rounded':''"><div class="text-[10px] text-slate-400">配息年</div><div class="font-bold text-sm text-amber-600" x-text="stock.cons_div"></div></div>
                    </div>
                </div>
            </template>
            <div x-show="displayCount < filteredStocks.length" class="text-center py-4"><button @click="displayCount+=20" class="bg-white border border-slate-300 text-slate-600 px-6 py-2 rounded-full text-sm font-bold shadow-sm hover:bg-slate-50 transition-all">載入更多... (<span x-text="filteredStocks.length-displayCount"></span>)</button></div>
        </div>
    </main>
    <script>
        function app() {{
            return {{
                stocks: {json_db}, filters: [], newFilter: {{ type: 'yield_avg', operator: '>=', value: 5 }}, showFilter: true, sortKey: 'yield', sortDesc: true, displayCount: 20,
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
                getLabel(f) {{ const map = {{ 'roe_avg': 'ROE', 'eps_ttm': 'EPS', 'eps_avg': '5年EPS', 'gross_margin': '毛利率', 'yield': '殖利率', 'yield_avg': '5年殖利', 'pe': 'PE', 'pb': 'PB', 'rev_growth': '營收YoY', 'vol': '成交量', 'ma_bull': '站上月線', 'cons_div': '連續配息' }}; return f.type === 'ma_bull' ? map[f.type] : `${{map[f.type]}} ${{f.operator}} ${{f.value}}`; }},
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
