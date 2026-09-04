// --- DOM Elements ---
const dom = {
    ticker: document.getElementById('tickerSelect'),
    dataSource: document.getElementById('dataSourceSelect'),
    dataPath: document.getElementById('dataPathInput'),
    model: document.getElementById('modelSelect'),
    timeframe: document.getElementById('timeframeSelect'),
    forecast: document.getElementById('forecastSelect'),
    history: document.getElementById('historySelect'),
    btnRun: document.getElementById('btnRun'),
    btnRetrain: document.getElementById('btnRetrain'),
    btnStop: document.getElementById('btnStop'),
    chkCandles: document.getElementById('chkCandles'),
    chkPrediction: document.getElementById('chkPrediction'),
    chkEma9: document.getElementById('chkEma9'),
    chkEma20: document.getElementById('chkEma20'),
    chkMa: document.getElementById('chkMa'),
    chkVwap: document.getElementById('chkVwap'),
    chkRsi: document.getElementById('chkRsi'),
    chkVolumeDelta: document.getElementById('chkVolumeDelta'),
    chkStochastic: document.getElementById('chkStochastic'),
    chkMfi: document.getElementById('chkMfi'),
    chkBollinger: document.getElementById('chkBollinger'),
    chkObv: document.getElementById('chkObv'),
    chkAdx: document.getElementById('chkAdx'),
    chkMacd: document.getElementById('chkMacd'),
    // Training Indicators panel - separate checkboxes from the chart
    // display panel above. These are the ONLY ones that should ever be
    // sent to /train, since they map 1:1 onto GRU_Training.py's
    // INDICATOR_COLUMN_MAP (EMA9, EMA20, VWAP, RSI, Stochastic).
    tiEma9: document.getElementById('tiEma9'),
    tiEma20: document.getElementById('tiEma20'),
    tiVwap: document.getElementById('tiVwap'),
    tiRsi: document.getElementById('tiRsi'),
    tiStochastic: document.getElementById('tiStochastic'),
    statusBar: document.getElementById('statusBar')
};

// --- State Variables ---
const TABS = [0, 1, 2, 3];
let activeTab = 1; // Chart Analysis 1 is active by default
let syncingCharts = false; // guards against infinite sync loops
let pollingInterval = null;

// Map to hold charts and series objects per tab
const tabCharts = {};
const tabSeries = {};

// Helper to get DOM containers for a specific tab
function getTabDom(t) {
    return {
        chartContainer: document.getElementById(`chartContainer${t}`),
        rsiContainer: document.getElementById(`rsiContainer${t}`),
        volumeDeltaContainer: document.getElementById(`volumeDeltaContainer${t}`),
        macdContainer: document.getElementById(`macdContainer${t}`),
        stochasticContainer: document.getElementById(`stochasticContainer${t}`),
        mfiContainer: document.getElementById(`mfiContainer${t}`),
        adxContainer: document.getElementById(`adxContainer${t}`),
        obvContainer: document.getElementById(`obvContainer${t}`)
    };
}

// All secondary (sub) charts for a specific tab, used for resize/sync loops
function subCharts(t) {
    const tc = tabCharts[t];
    if (!tc) return [];
    return [tc.rsiChart, tc.volumeDeltaChart, tc.macdChart, tc.stochasticChart, tc.mfiChart, tc.adxChart, tc.obvChart];
}

// Defensive filter: lightweight-charts throws an uncaught
// "Value is null" error if any data point has a null/undefined
// value (or a null OHLC field for candles).
function cleanSeriesData(arr, isCandle = false) {
    if (!Array.isArray(arr)) return [];
    return arr.filter(point => {
        if (point == null || point.time == null) return false;
        if (isCandle) {
            return point.open != null && point.high != null &&
                point.low != null && point.close != null &&
                !Number.isNaN(point.open) && !Number.isNaN(point.high) &&
                !Number.isNaN(point.low) && !Number.isNaN(point.close);
        }
        return point.value != null && !Number.isNaN(point.value);
    });
}

// --- Initialization ---
async function init() {
    TABS.forEach(t => {
        tabCharts[t] = {};
        tabSeries[t] = {};
        initTabCharts(t);
        syncChartTimeScales(t);
    });

    await fetchDropdowns();
    setupEventListeners();

    setStatus("Ready. Configure your settings and click 'Run Inference' to begin.");
}

function initTabCharts(t) {
    const containers = getTabDom(t);
    const tc = tabCharts[t];
    const ts = tabSeries[t];

    // Main Chart
    tc.chart = LightweightCharts.createChart(containers.chartContainer, {
        layout: {
            background: { type: 'solid', color: '#1e1e24' },
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: { color: '#2b2b36' },
            horzLines: { color: '#2b2b36' },
        },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: '#2b2b36' },
        timeScale: {
            borderColor: '#2b2b36',
            timeVisible: true,
            secondsVisible: false
        },
    });

    // Create Series
    ts.candles = tc.chart.addSeries(LightweightCharts.CandlestickSeries, {
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
    });
    ts.candleMarkers = LightweightCharts.createSeriesMarkers(ts.candles);

    ts.volume = tc.chart.addSeries(LightweightCharts.HistogramSeries, {
        color: '#26a69a',
        priceFormat: {
            type: 'volume',
        },
        priceScaleId: 'volume_scale',
    });

    tc.chart.priceScale('volume_scale').applyOptions({
        scaleMargins: {
            top: 0.8,
            bottom: 0,
        },
    });

    ts.ema9 = tc.chart.addSeries(LightweightCharts.LineSeries, { color: '#2962FF', lineWidth: 1, crosshairMarkerVisible: false });
    ts.ema20 = tc.chart.addSeries(LightweightCharts.LineSeries, { color: '#FF9800', lineWidth: 1, crosshairMarkerVisible: false });
    ts.ma = tc.chart.addSeries(LightweightCharts.LineSeries, { color: '#4CAF50', lineWidth: 1, crosshairMarkerVisible: false });
    ts.vwap = tc.chart.addSeries(LightweightCharts.LineSeries, { color: '#E91E63', lineWidth: 1, lineStyle: 2, crosshairMarkerVisible: false });

    // Bollinger Bands: 3 price-scale overlay lines (upper / middle / lower)
    ts.bbUpper = tc.chart.addSeries(LightweightCharts.LineSeries, { color: '#9C27B0', lineWidth: 1, lineStyle: 2, crosshairMarkerVisible: false });
    ts.bbMiddle = tc.chart.addSeries(LightweightCharts.LineSeries, { color: '#9C27B0', lineWidth: 1, crosshairMarkerVisible: false });
    ts.bbLower = tc.chart.addSeries(LightweightCharts.LineSeries, { color: '#9C27B0', lineWidth: 1, lineStyle: 2, crosshairMarkerVisible: false });

    ts.historical = tc.chart.addSeries(LightweightCharts.LineSeries, { color: '#B30DD6', lineWidth: 2, lineStyle: 1 }); // Purple, dashed
    ts.future = tc.chart.addSeries(LightweightCharts.LineSeries, { color: '#FFD700', lineWidth: 2 }); // Gold

    // RSI
    tc.rsiChart = LightweightCharts.createChart(containers.rsiContainer, {
        layout: {
            background: { type: 'solid', color: '#1e1e24' },
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: { color: '#2b2b36' },
            horzLines: { color: '#2b2b36' },
        },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: {
            borderColor: '#2b2b36',
            scaleMargins: { top: 0.15, bottom: 0.15 },
        },
        timeScale: {
            borderColor: '#2b2b36',
            timeVisible: true,
            secondsVisible: false,
        },
    });

    ts.rsi = tc.rsiChart.addSeries(LightweightCharts.LineSeries, {
        color: '#FFEB3B',
        lineWidth: 1.5,
        crosshairMarkerVisible: true,
        priceLineVisible: false,
        lastValueVisible: true,
        autoscaleInfoProvider: () => ({
            priceRange: { minValue: 0, maxValue: 100 },
        }),
    });

    ts.rsi.createPriceLine({
        price: 70,
        color: '#787b86',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: true,
        title: 'Overbought',
    });
    ts.rsi.createPriceLine({
        price: 30,
        color: '#787b86',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: true,
        title: 'Oversold',
    });

    // Volume Delta
    tc.volumeDeltaChart = LightweightCharts.createChart(containers.volumeDeltaContainer, {
        layout: {
            background: { type: 'solid', color: '#1e1e24' },
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: { color: '#2b2b36' },
            horzLines: { color: '#2b2b36' },
        },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: {
            borderColor: '#2b2b36',
            scaleMargins: { top: 0.15, bottom: 0.15 },
        },
        timeScale: {
            borderColor: '#2b2b36',
            timeVisible: true,
            secondsVisible: false,
        },
    });

    ts.volumeDelta = tc.volumeDeltaChart.addSeries(LightweightCharts.HistogramSeries, {
        priceLineVisible: false,
        lastValueVisible: true,
        base: 0,
    });

    ts.volumeDelta.createPriceLine({
        price: 0,
        color: '#787b86',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Solid,
        axisLabelVisible: false,
    });

    // MACD
    tc.macdChart = LightweightCharts.createChart(containers.macdContainer, {
        layout: {
            background: { type: 'solid', color: '#1e1e24' },
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: { color: '#2b2b36' },
            horzLines: { color: '#2b2b36' },
        },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: {
            borderColor: '#2b2b36',
            scaleMargins: { top: 0.15, bottom: 0.15 },
        },
        timeScale: {
            borderColor: '#2b2b36',
            timeVisible: true,
            secondsVisible: false,
        },
    });

    ts.macdHistogram = tc.macdChart.addSeries(LightweightCharts.HistogramSeries, {
        priceLineVisible: false,
        lastValueVisible: false,
        base: 0,
    });
    ts.macdLine = tc.macdChart.addSeries(LightweightCharts.LineSeries, {
        color: '#2962FF',
        lineWidth: 1.5,
        priceLineVisible: false,
    });
    ts.macdSignal = tc.macdChart.addSeries(LightweightCharts.LineSeries, {
        color: '#FF9800',
        lineWidth: 1.5,
        priceLineVisible: false,
    });

    ts.macdHistogram.createPriceLine({
        price: 0,
        color: '#787b86',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Solid,
        axisLabelVisible: false,
    });

    // Stochastic
    tc.stochasticChart = LightweightCharts.createChart(containers.stochasticContainer, {
        layout: {
            background: { type: 'solid', color: '#1e1e24' },
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: { color: '#2b2b36' },
            horzLines: { color: '#2b2b36' },
        },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: {
            borderColor: '#2b2b36',
            scaleMargins: { top: 0.15, bottom: 0.15 },
        },
        timeScale: {
            borderColor: '#2b2b36',
            timeVisible: true,
            secondsVisible: false,
        },
    });

    ts.stochK = tc.stochasticChart.addSeries(LightweightCharts.LineSeries, {
        color: '#00BCD4',
        lineWidth: 1.5,
        priceLineVisible: false,
        autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 100 } }),
    });
    ts.stochD = tc.stochasticChart.addSeries(LightweightCharts.LineSeries, {
        color: '#E91E63',
        lineWidth: 1.5,
        priceLineVisible: false,
    });

    ts.stochK.createPriceLine({ price: 80, color: '#787b86', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: 'Overbought' });
    ts.stochK.createPriceLine({ price: 20, color: '#787b86', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: 'Oversold' });

    // MFI
    tc.mfiChart = LightweightCharts.createChart(containers.mfiContainer, {
        layout: {
            background: { type: 'solid', color: '#1e1e24' },
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: { color: '#2b2b36' },
            horzLines: { color: '#2b2b36' },
        },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: {
            borderColor: '#2b2b36',
            scaleMargins: { top: 0.15, bottom: 0.15 },
        },
        timeScale: {
            borderColor: '#2b2b36',
            timeVisible: true,
            secondsVisible: false,
        },
    });

    ts.mfi = tc.mfiChart.addSeries(LightweightCharts.LineSeries, {
        color: '#8BC34A',
        lineWidth: 1.5,
        priceLineVisible: false,
        autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 100 } }),
    });

    ts.mfi.createPriceLine({ price: 80, color: '#787b86', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: 'Overbought' });
    ts.mfi.createPriceLine({ price: 20, color: '#787b86', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: 'Oversold' });

    // ADX
    tc.adxChart = LightweightCharts.createChart(containers.adxContainer, {
        layout: {
            background: { type: 'solid', color: '#1e1e24' },
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: { color: '#2b2b36' },
            horzLines: { color: '#2b2b36' },
        },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: {
            borderColor: '#2b2b36',
            scaleMargins: { top: 0.15, bottom: 0.15 },
        },
        timeScale: {
            borderColor: '#2b2b36',
            timeVisible: true,
            secondsVisible: false,
        },
    });

    ts.adx = tc.adxChart.addSeries(LightweightCharts.LineSeries, {
        color: '#FF5722',
        lineWidth: 1.5,
        priceLineVisible: false,
        autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 100 } }),
    });

    ts.adx.createPriceLine({ price: 25, color: '#787b86', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: 'Trend Threshold' });

    // OBV
    tc.obvChart = LightweightCharts.createChart(containers.obvContainer, {
        layout: {
            background: { type: 'solid', color: '#1e1e24' },
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: { color: '#2b2b36' },
            horzLines: { color: '#2b2b36' },
        },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: {
            borderColor: '#2b2b36',
            scaleMargins: { top: 0.15, bottom: 0.15 },
        },
        timeScale: {
            borderColor: '#2b2b36',
            timeVisible: true,
            secondsVisible: false,
        },
    });

    ts.obv = tc.obvChart.addSeries(LightweightCharts.LineSeries, {
        color: '#03A9F4',
        lineWidth: 1.5,
        priceLineVisible: false,
    });
}

// Keep all charts scrolling/zooming together per tab
function syncChartTimeScales(t) {
    const tc = tabCharts[t];
    if (!tc) return;
    const all = [tc.chart, ...subCharts(t)];

    all.forEach((c) => {
        if (!c) return;
        c.timeScale().subscribeVisibleLogicalRangeChange((range) => {
            if (syncingCharts || !range) return;
            syncingCharts = true;
            all.forEach((other) => {
                if (other && other !== c) other.timeScale().setVisibleLogicalRange(range);
            });
            syncingCharts = false;
        });
    });
}

function resizeCharts() {
    TABS.forEach(t => {
        const tc = tabCharts[t];
        if (!tc || !tc.chart) return;
        const containers = getTabDom(t);
        const panel = document.getElementById(`chartAnalysis${t}`);
        if (panel && panel.classList.contains('active')) {
            tc.chart.resize(containers.chartContainer.clientWidth, containers.chartContainer.clientHeight);
            const panels = [
                [containers.rsiContainer, tc.rsiChart],
                [containers.volumeDeltaContainer, tc.volumeDeltaChart],
                [containers.macdContainer, tc.macdChart],
                [containers.stochasticContainer, tc.stochasticChart],
                [containers.mfiContainer, tc.mfiChart],
                [containers.adxContainer, tc.adxChart],
                [containers.obvContainer, tc.obvChart],
            ];
            panels.forEach(([container, chartInstance]) => {
                if (container.style.display !== 'none' && chartInstance) {
                    chartInstance.resize(container.clientWidth, container.clientHeight);
                }
            });
        }
    });
}

async function fetchDropdowns() {
    try {
        const [symRes, modRes] = await Promise.all([fetch('/symbols'), fetch('/models')]);

        if (!symRes.ok || !modRes.ok) {
            throw new Error(`HTTP error! Status: ${symRes.status}`);
        }

        const symbols = await symRes.json();
        const models = await modRes.json();

        dom.ticker.innerHTML = symbols.map(s => `<option value="${s}">${s}</option>`).join('');
        dom.model.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join('');

        if (symbols.includes("SPY")) dom.ticker.value = "SPY";
        if (models.includes("TCN")) dom.model.value = "TCN";

    } catch (err) {
        console.error("Dropdown Fetch Error:", err);
        setStatus("Error loading startup configuration. Check console.");
    }
}

function setupEventListeners() {
    dom.btnRun.addEventListener('click', runForecast);
    dom.btnRetrain.addEventListener('click', startTraining);
    dom.btnStop.addEventListener('click', stopTraining);

    // Visibility toggles applied across all tabs for consistency
    if (dom.chkCandles) {
        dom.chkCandles.addEventListener('change', (e) => {
            TABS.forEach(t => {
                tabSeries[t].candles && tabSeries[t].candles.applyOptions({ visible: e.target.checked });
                tabSeries[t].volume && tabSeries[t].volume.applyOptions({ visible: e.target.checked });
            });
        });
    }
    if (dom.chkPrediction) {
        dom.chkPrediction.addEventListener('change', (e) => {
            TABS.forEach(t => {
                tabSeries[t].historical && tabSeries[t].historical.applyOptions({ visible: e.target.checked });
                tabSeries[t].future && tabSeries[t].future.applyOptions({ visible: e.target.checked });
            });
        });
    }
    if (dom.chkEma9) {
        dom.chkEma9.addEventListener('change', (e) => {
            TABS.forEach(t => tabSeries[t].ema9 && tabSeries[t].ema9.applyOptions({ visible: e.target.checked }));
        });
    }
    if (dom.chkEma20) {
        dom.chkEma20.addEventListener('change', (e) => {
            TABS.forEach(t => tabSeries[t].ema20 && tabSeries[t].ema20.applyOptions({ visible: e.target.checked }));
        });
    }
    if (dom.chkMa) {
        dom.chkMa.addEventListener('change', (e) => {
            TABS.forEach(t => tabSeries[t].ma && tabSeries[t].ma.applyOptions({ visible: e.target.checked }));
        });
    }
    if (dom.chkVwap) {
        dom.chkVwap.addEventListener('change', (e) => {
            TABS.forEach(t => tabSeries[t].vwap && tabSeries[t].vwap.applyOptions({ visible: e.target.checked }));
        });
    }
    if (dom.chkRsi) {
        dom.chkRsi.addEventListener('change', (e) => {
            TABS.forEach(t => toggleSubPanel(getTabDom(t).rsiContainer, tabCharts[t].rsiChart, e.target.checked));
        });
    }
    if (dom.chkVolumeDelta) {
        dom.chkVolumeDelta.addEventListener('change', (e) => {
            TABS.forEach(t => toggleSubPanel(getTabDom(t).volumeDeltaContainer, tabCharts[t].volumeDeltaChart, e.target.checked));
        });
    }
    if (dom.chkBollinger) {
        dom.chkBollinger.addEventListener('change', (e) => {
            TABS.forEach(t => {
                tabSeries[t].bbUpper && tabSeries[t].bbUpper.applyOptions({ visible: e.target.checked });
                tabSeries[t].bbMiddle && tabSeries[t].bbMiddle.applyOptions({ visible: e.target.checked });
                tabSeries[t].bbLower && tabSeries[t].bbLower.applyOptions({ visible: e.target.checked });
            });
        });
    }
    if (dom.chkMacd) {
        dom.chkMacd.addEventListener('change', (e) => {
            TABS.forEach(t => toggleSubPanel(getTabDom(t).macdContainer, tabCharts[t].macdChart, e.target.checked));
        });
    }
    if (dom.chkStochastic) {
        dom.chkStochastic.addEventListener('change', (e) => {
            TABS.forEach(t => toggleSubPanel(getTabDom(t).stochasticContainer, tabCharts[t].stochasticChart, e.target.checked));
        });
    }
    if (dom.chkMfi) {
        dom.chkMfi.addEventListener('change', (e) => {
            TABS.forEach(t => toggleSubPanel(getTabDom(t).mfiContainer, tabCharts[t].mfiChart, e.target.checked));
        });
    }
    if (dom.chkAdx) {
        dom.chkAdx.addEventListener('change', (e) => {
            TABS.forEach(t => toggleSubPanel(getTabDom(t).adxContainer, tabCharts[t].adxChart, e.target.checked));
        });
    }
    if (dom.chkObv) {
        dom.chkObv.addEventListener('change', (e) => {
            TABS.forEach(t => toggleSubPanel(getTabDom(t).obvContainer, tabCharts[t].obvChart, e.target.checked));
        });
    }

    [dom.timeframe, dom.history, dom.forecast].forEach(el => {
        el.addEventListener('change', runForecast);
    });

    [dom.ticker, dom.model].forEach(el => {
        el.addEventListener('change', () => {
            dom.btnRun.disabled = false;
            setStatus("Configuration changed. Click 'Run Inference' to generate new charts.");
        });
    });

    // Custom tab event handling to track the active tab index and resize charts
    document.querySelectorAll('.chart-analysis-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.getAttribute('data-target');
            activeTab = parseInt(targetId.replace('chartAnalysis', ''));
            setTimeout(resizeCharts, 50);
        });
    });

    window.addEventListener('resize', resizeCharts);
}

// Shows/hides a whole sub-panel chart
function toggleSubPanel(container, chartInstance, visible) {
    if (container) {
        container.style.display = visible ? 'block' : 'none';
    }
    requestAnimationFrame(resizeCharts);
}

function setStatus(text) {
    dom.statusBar.innerText = text;
}

// Collects the checked boxes from the "Training Indicators" panel only.
function getSelectedTrainingIndicators() {
    const map = {
        tiEma9: "EMA9",
        tiEma20: "EMA20",
        tiVwap: "VWAP",
        tiRsi: "RSI",
        tiStochastic: "Stochastic"
    };

    return Object.entries(map)
        .filter(([id]) => dom[id] && dom[id].checked)
        .map(([, name]) => name);
}

async function runForecast() {
    setStatus("Fetching predictions...");
    dom.btnRun.disabled = true;

    try {
        let historyVal = dom.history.value;
        historyVal = (historyVal === "All") ? 100000 : parseInt(historyVal);

        const payload = {
            symbol: dom.ticker.value,
            model: dom.model.value,
            data_source: dom.dataSource ? dom.dataSource.value : "ftp",
            data_path: dom.dataPath ? dom.dataPath.value : "",
            timeframe: parseInt(dom.timeframe.value),
            forecast_steps: parseInt(dom.forecast.value),
            history: historyVal,
            show_ema9: dom.chkEma9 ? dom.chkEma9.checked : false,
            show_ema20: dom.chkEma20 ? dom.chkEma20.checked : false,
            show_ma: dom.chkMa ? dom.chkMa.checked : false,
            show_vwap: dom.chkVwap ? dom.chkVwap.checked : false,
            show_rsi: dom.chkRsi ? dom.chkRsi.checked : false,
            show_volume_delta: dom.chkVolumeDelta ? dom.chkVolumeDelta.checked : false,
            show_bollinger: dom.chkBollinger ? dom.chkBollinger.checked : false,
            show_macd: dom.chkMacd ? dom.chkMacd.checked : false,
            show_stochastic: dom.chkStochastic ? dom.chkStochastic.checked : false,
            show_mfi: dom.chkMfi ? dom.chkMfi.checked : false,
            show_adx: dom.chkAdx ? dom.chkAdx.checked : false,
            show_obv: dom.chkObv ? dom.chkObv.checked : false
        };

        const res = await fetch('/forecast', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();

        const ts = tabSeries[activeTab];
        const tc = tabCharts[activeTab];

        if (!ts || !tc) return;

        ts.candles.setData(cleanSeriesData(data.candles, true));

        if (data.candles && data.candles.length > 0) {
            const volumeData = data.candles.map(point => ({
                time: point.time,
                value: point.volume || 0,
                color: point.close >= point.open ? '#26a69a' : '#ef5350'
            }));
            ts.volume.setData(cleanSeriesData(volumeData));
        } else {
            ts.volume.setData([]);
        }
        ts.ema9.setData(cleanSeriesData(data.ema9));
        ts.ema20.setData(cleanSeriesData(data.ema20));
        ts.ma.setData(cleanSeriesData(data.ma));
        ts.vwap.setData(cleanSeriesData(data.vwap));
        ts.historical.setData(cleanSeriesData(data.historical_predictions));
        ts.future.setData(cleanSeriesData(data.future_predictions));

        if (data.signals && data.signals.length > 0) {
            ts.candleMarkers.setMarkers(data.signals);
        } else {
            ts.candleMarkers.setMarkers([]);
        }

        if (data.rsi && data.rsi.length > 0) {
            ts.rsi.setData(cleanSeriesData(data.rsi));
        } else {
            console.warn("No RSI data returned from /forecast — check backend response.");
        }

        if (data.volume_delta && data.volume_delta.length > 0) {
            ts.volumeDelta.setData(cleanSeriesData(data.volume_delta));
        } else {
            console.warn("No Volume Delta data returned from /forecast — check backend response.");
        }

        if (data.bollinger_upper && data.bollinger_upper.length > 0) {
            ts.bbUpper.setData(cleanSeriesData(data.bollinger_upper));
            ts.bbMiddle.setData(cleanSeriesData(data.bollinger_middle));
            ts.bbLower.setData(cleanSeriesData(data.bollinger_lower));
        } else {
            console.warn("No Bollinger Bands data returned from /forecast — check backend response.");
        }

        if (data.macd_line && data.macd_line.length > 0) {
            ts.macdLine.setData(cleanSeriesData(data.macd_line));
            ts.macdSignal.setData(cleanSeriesData(data.macd_signal));
            ts.macdHistogram.setData(cleanSeriesData(data.macd_histogram));
        } else {
            console.warn("No MACD data returned from /forecast — check backend response.");
        }

        if (data.stochastic_k && data.stochastic_k.length > 0) {
            ts.stochK.setData(cleanSeriesData(data.stochastic_k));
            ts.stochD.setData(cleanSeriesData(data.stochastic_d));
        } else {
            console.warn("No Stochastic data returned from /forecast — check backend response.");
        }

        if (data.mfi && data.mfi.length > 0) {
            ts.mfi.setData(cleanSeriesData(data.mfi));
        } else {
            console.warn("No MFI data returned from /forecast — check backend response.");
        }

        if (data.adx && data.adx.length > 0) {
            ts.adx.setData(cleanSeriesData(data.adx));
        } else {
            console.warn("No ADX data returned from /forecast — check backend response.");
        }

        if (data.obv && data.obv.length > 0) {
            ts.obv.setData(cleanSeriesData(data.obv));
        } else {
            console.warn("No OBV data returned from /forecast — check backend response.");
        }

        ts.candles.applyOptions({ visible: dom.chkCandles.checked });
        ts.volume.applyOptions({ visible: dom.chkCandles.checked });
        ts.ema9.applyOptions({ visible: dom.chkEma9.checked });
        ts.ema20.applyOptions({ visible: dom.chkEma20.checked });
        ts.ma.applyOptions({ visible: dom.chkMa ? dom.chkMa.checked : false });
        ts.vwap.applyOptions({ visible: dom.chkVwap.checked });
        ts.bbUpper.applyOptions({ visible: dom.chkBollinger.checked });
        ts.bbMiddle.applyOptions({ visible: dom.chkBollinger.checked });
        ts.bbLower.applyOptions({ visible: dom.chkBollinger.checked });
        ts.historical.applyOptions({ visible: dom.chkPrediction.checked });
        ts.future.applyOptions({ visible: dom.chkPrediction.checked });

        const containers = getTabDom(activeTab);
        toggleSubPanel(containers.rsiContainer, tc.rsiChart, dom.chkRsi.checked);
        toggleSubPanel(containers.volumeDeltaContainer, tc.volumeDeltaChart, dom.chkVolumeDelta ? dom.chkVolumeDelta.checked : false);
        toggleSubPanel(containers.macdContainer, tc.macdChart, dom.chkMacd.checked);
        toggleSubPanel(containers.stochasticContainer, tc.stochasticChart, dom.chkStochastic.checked);
        toggleSubPanel(containers.mfiContainer, tc.mfiChart, dom.chkMfi ? dom.chkMfi.checked : false);
        toggleSubPanel(containers.adxContainer, tc.adxChart, dom.chkAdx.checked);
        toggleSubPanel(containers.obvContainer, tc.obvChart, dom.chkObv.checked);

        tc.chart.timeScale().fitContent();
        subCharts(activeTab).forEach((c) => c && c.timeScale().fitContent());
        setStatus(`Status: Ready | Loaded ${data.candles.length} candles for ${dom.ticker.value} on Tab ${activeTab}`);

    } catch (err) {
        setStatus(`Error: ${err.message}`);
        dom.btnRun.disabled = false;
    }
}

async function startTraining() {
    try {
        const payload = {
            symbol: dom.ticker.value,
            model: dom.model.value,
            data_source: dom.dataSource ? dom.dataSource.value : "ftp",
            data_path: dom.dataPath ? dom.dataPath.value : "",
            indicators: getSelectedTrainingIndicators()
        };
        const res = await fetch('/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (data.success) {
            dom.btnRetrain.disabled = true;
            dom.btnRun.style.display = 'none';
            dom.btnStop.style.display = 'inline-block';
            startPolling();
        } else {
            alert(data.message);
        }
    } catch (err) {
        setStatus(`Training Error: ${err.message}`);
    }
}

async function stopTraining() {
    await fetch('/stop-training', { method: 'POST' });
    setStatus("Stopping training...");
}

function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);

    pollingInterval = setInterval(async () => {
        try {
            const res = await fetch('/training-status');
            const state = await res.json();

            if (!state.running) {
                clearInterval(pollingInterval);

                dom.btnRetrain.disabled = false;
                dom.btnStop.style.display = 'none';
                dom.btnRun.style.display = 'inline-block';

                if (state.epoch > 0) {
                    setStatus(`Training Complete for ${state.model}. Auto-refreshing forecast...`);
                    await runForecast();
                } else {
                    setStatus("Training stopped by user.");
                    dom.btnRun.disabled = false;
                }
                return;
            }

            setStatus(
                `Training ${state.model} [${state.symbol}] | Epoch: ${state.epoch}/${state.total_epochs} | Train Loss: ${state.train_loss.toFixed(6)} | Val Loss: ${state.val_loss.toFixed(6)}`
            );
        } catch (err) {
            console.error("Polling error", err);
        }
    }, 1000);
}

// Bootstrap
document.addEventListener('DOMContentLoaded', init);