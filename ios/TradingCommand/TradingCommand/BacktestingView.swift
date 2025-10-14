import SwiftUI
import Charts

struct BacktestingView: View {
    @StateObject private var backtestManager = BacktestManager()
    @State private var selectedStrategy = "Momentum"
    @State private var showStrategyPicker = false
    @State private var showDateRangePicker = false
    @State private var isRunningBacktest = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Strategy Selection
                    StrategySelectionCard(
                        selectedStrategy: $selectedStrategy,
                        showPicker: $showStrategyPicker
                    )
                    
                    // Backtest Configuration
                    BacktestConfigurationCard(
                        manager: backtestManager,
                        showDatePicker: $showDateRangePicker
                    )
                    
                    // Run Backtest Button
                    RunBacktestButton(
                        isRunning: $isRunningBacktest,
                        action: runBacktest
                    )
                    
                    if backtestManager.hasResults {
                        // Results Summary
                        BacktestResultsSummary(results: backtestManager.results)
                        
                        // Performance Charts
                        BacktestChartsSection(results: backtestManager.results)
                        
                        // Trade Analysis
                        TradeAnalysisSection(trades: backtestManager.trades)
                        
                        // Risk Metrics
                        RiskMetricsSection(metrics: backtestManager.riskMetrics)
                        
                        // Monthly Returns Heatmap
                        MonthlyReturnsHeatmap(returns: backtestManager.monthlyReturns)
                    }
                }
                .padding()
            }
            .navigationTitle("Backtesting")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Button(action: exportResults) {
                            Label("Export PDF", systemImage: "doc.fill")
                        }
                        Button(action: shareResults) {
                            Label("Share", systemImage: "square.and.arrow.up")
                        }
                        Button(action: saveStrategy) {
                            Label("Save Strategy", systemImage: "bookmark.fill")
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
        }
        .sheet(isPresented: $showStrategyPicker) {
            StrategyPickerSheet(selectedStrategy: $selectedStrategy)
        }
        .sheet(isPresented: $showDateRangePicker) {
            DateRangePickerSheet(manager: backtestManager)
        }
    }
    
    func runBacktest() {
        isRunningBacktest = true
        
        Task {
            await backtestManager.runBacktest(strategy: selectedStrategy)
            await MainActor.run {
                isRunningBacktest = false
            }
        }
    }
    
    func exportResults() {
        // Export to PDF
    }
    
    func shareResults() {
        // Share results
    }
    
    func saveStrategy() {
        // Save strategy configuration
    }
}

// MARK: - Strategy Selection Card
struct StrategySelectionCard: View {
    @Binding var selectedStrategy: String
    @Binding var showPicker: Bool
    
    let strategies = [
        ("Momentum", "Trend following with momentum indicators", Color.blue),
        ("Mean Reversion", "Buy dips, sell rallies", Color.green),
        ("Arbitrage", "Cross-exchange opportunities", Color.purple),
        ("ML Prediction", "Neural network signals", Color.orange),
        ("Custom", "Your custom strategy", Color.pink)
    ]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Strategy")
                .font(.headline)
            
            Button(action: { showPicker = true }) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(selectedStrategy)
                            .font(.subheadline)
                            .fontWeight(.medium)
                        
                        if let strategy = strategies.first(where: { $0.0 == selectedStrategy }) {
                            Text(strategy.1)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    Spacer()
                    
                    Image(systemName: "chevron.down")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(8)
            }
            .buttonStyle(PlainButtonStyle())
            
            // Quick Strategy Stats
            HStack(spacing: 16) {
                QuickStat(label: "Win Rate", value: "68%", color: .green)
                QuickStat(label: "Sharpe", value: "1.85", color: .blue)
                QuickStat(label: "Max DD", value: "-12%", color: .red)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct QuickStat: View {
    let label: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(value)
                .font(.caption)
                .fontWeight(.bold)
                .foregroundColor(color)
        }
    }
}

// MARK: - Backtest Configuration Card
struct BacktestConfigurationCard: View {
    @ObservedObject var manager: BacktestManager
    @Binding var showDatePicker: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Configuration")
                .font(.headline)
            
            // Date Range
            VStack(alignment: .leading, spacing: 8) {
                Text("Date Range")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Button(action: { showDatePicker = true }) {
                    HStack {
                        Image(systemName: "calendar")
                            .foregroundColor(.blue)
                        Text("\(manager.startDate, formatter: dateFormatter) - \(manager.endDate, formatter: dateFormatter)")
                            .font(.subheadline)
                        Spacer()
                        Text("\(daysBetween) days")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    .background(Color(.systemGray5))
                    .cornerRadius(8)
                }
                .buttonStyle(PlainButtonStyle())
            }
            
            // Capital & Risk Settings
            HStack(spacing: 16) {
                SettingField(
                    label: "Initial Capital",
                    value: manager.initialCapital,
                    formatter: currencyFormatter
                )
                
                SettingField(
                    label: "Position Size",
                    value: manager.positionSize,
                    formatter: percentFormatter
                )
            }
            
            HStack(spacing: 16) {
                SettingField(
                    label: "Stop Loss",
                    value: manager.stopLoss,
                    formatter: percentFormatter
                )
                
                SettingField(
                    label: "Take Profit",
                    value: manager.takeProfit,
                    formatter: percentFormatter
                )
            }
            
            // Advanced Settings Toggle
            DisclosureGroup("Advanced Settings") {
                VStack(spacing: 12) {
                    Toggle("Include Slippage", isOn: $manager.includeSlippage)
                    Toggle("Include Fees", isOn: $manager.includeFees)
                    
                    HStack {
                        Text("Fee Rate")
                        Spacer()
                        Text("\(manager.feeRate * 100, specifier: "%.2f")%")
                            .foregroundColor(.secondary)
                    }
                    .font(.subheadline)
                    
                    Slider(value: $manager.feeRate, in: 0...0.01)
                        .accentColor(.blue)
                }
                .padding(.top, 8)
            }
            .font(.subheadline)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    var daysBetween: Int {
        Calendar.current.dateComponents([.day], from: manager.startDate, to: manager.endDate).day ?? 0
    }
    
    var dateFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return formatter
    }
    
    var currencyFormatter: NumberFormatter {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.maximumFractionDigits = 0
        return formatter
    }
    
    var percentFormatter: NumberFormatter {
        let formatter = NumberFormatter()
        formatter.numberStyle = .percent
        formatter.maximumFractionDigits = 1
        return formatter
    }
}

struct SettingField: View {
    let label: String
    let value: Double
    let formatter: NumberFormatter
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(formatter.string(from: NSNumber(value: value)) ?? "")
                .font(.subheadline)
                .fontWeight(.medium)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(8)
        .background(Color(.systemGray5))
        .cornerRadius(6)
    }
}

// MARK: - Run Backtest Button
struct RunBacktestButton: View {
    @Binding var isRunning: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack {
                if isRunning {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        .scaleEffect(0.8)
                } else {
                    Image(systemName: "play.fill")
                }
                Text(isRunning ? "Running Backtest..." : "Run Backtest")
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(
                LinearGradient(
                    colors: [Color.blue, Color.blue.opacity(0.8)],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )
            .foregroundColor(.white)
            .cornerRadius(12)
            .disabled(isRunning)
        }
    }
}

// MARK: - Backtest Results Summary
struct BacktestResultsSummary: View {
    let results: BacktestResults
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Results")
                    .font(.headline)
                
                Spacer()
                
                ResultBadge(
                    text: results.isProfitable ? "Profitable" : "Loss",
                    color: results.isProfitable ? .green : .red
                )
            }
            
            // Key Metrics Grid
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 12) {
                ResultMetric(
                    label: "Total Return",
                    value: formatPercent(results.totalReturn),
                    trend: results.totalReturn > 0 ? .up : .down
                )
                
                ResultMetric(
                    label: "Annual Return",
                    value: formatPercent(results.annualReturn),
                    trend: results.annualReturn > 0 ? .up : .down
                )
                
                ResultMetric(
                    label: "Win Rate",
                    value: formatPercent(results.winRate),
                    trend: results.winRate > 0.5 ? .up : .down
                )
                
                ResultMetric(
                    label: "Profit Factor",
                    value: String(format: "%.2f", results.profitFactor),
                    trend: results.profitFactor > 1 ? .up : .down
                )
                
                ResultMetric(
                    label: "Sharpe Ratio",
                    value: String(format: "%.2f", results.sharpeRatio),
                    trend: results.sharpeRatio > 1 ? .up : .down
                )
                
                ResultMetric(
                    label: "Max Drawdown",
                    value: formatPercent(results.maxDrawdown),
                    trend: .down
                )
            }
            
            // Final Capital
            HStack {
                Text("Final Capital")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                Spacer()
                Text(formatCurrency(results.finalCapital))
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(results.isProfitable ? .green : .red)
            }
            .padding(.top, 8)
        }
        .padding()
        .background(
            LinearGradient(
                colors: [
                    results.isProfitable ? Color.green.opacity(0.1) : Color.red.opacity(0.1),
                    Color(.systemGray6)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .cornerRadius(12)
    }
    
    func formatPercent(_ value: Double) -> String {
        String(format: "%+.2f%%", value * 100)
    }
    
    func formatCurrency(_ value: Double) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        return formatter.string(from: NSNumber(value: value)) ?? "$0"
    }
}

struct ResultBadge: View {
    let text: String
    let color: Color
    
    var body: some View {
        Text(text)
            .font(.caption)
            .fontWeight(.semibold)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(color.opacity(0.2))
            .foregroundColor(color)
            .cornerRadius(6)
    }
}

struct ResultMetric: View {
    let label: String
    let value: String
    let trend: Trend
    
    enum Trend {
        case up, down, neutral
        
        var color: Color {
            switch self {
            case .up: return .green
            case .down: return .red
            case .neutral: return .gray
            }
        }
        
        var icon: String {
            switch self {
            case .up: return "arrow.up.right"
            case .down: return "arrow.down.right"
            case .neutral: return "minus"
            }
        }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            
            HStack(spacing: 4) {
                Image(systemName: trend.icon)
                    .font(.caption)
                Text(value)
                    .font(.subheadline)
                    .fontWeight(.bold)
            }
            .foregroundColor(trend.color)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.systemGray5))
        .cornerRadius(8)
    }
}

// MARK: - Backtest Charts Section
struct BacktestChartsSection: View {
    let results: BacktestResults
    @State private var selectedChart = "Equity"
    let chartTypes = ["Equity", "Drawdown", "Returns", "Trades"]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Performance Charts")
                    .font(.headline)
                
                Spacer()
                
                Picker("Chart", selection: $selectedChart) {
                    ForEach(chartTypes, id: \.self) { type in
                        Text(type).tag(type)
                    }
                }
                .pickerStyle(MenuPickerStyle())
            }
            
            Group {
                switch selectedChart {
                case "Equity":
                    EquityCurveChart(data: results.equityCurve)
                case "Drawdown":
                    DrawdownChart(data: results.drawdownCurve)
                case "Returns":
                    ReturnsDistributionChart(data: results.returnsDistribution)
                case "Trades":
                    TradesChart(trades: results.trades)
                default:
                    EmptyView()
                }
            }
            .frame(height: 250)
            .padding()
            .background(Color(.systemGray5))
            .cornerRadius(8)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct EquityCurveChart: View {
    let data: [EquityPoint]
    
    var body: some View {
        Chart(data) { point in
            LineMark(
                x: .value("Date", point.date),
                y: .value("Equity", point.value)
            )
            .foregroundStyle(
                LinearGradient(
                    colors: [.blue, .green],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )
            .lineStyle(StrokeStyle(lineWidth: 2))
            
            AreaMark(
                x: .value("Date", point.date),
                y: .value("Equity", point.value)
            )
            .foregroundStyle(
                LinearGradient(
                    colors: [
                        Color.blue.opacity(0.3),
                        Color.blue.opacity(0.05)
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
        }
        .chartYAxis {
            AxisMarks(position: .leading)
        }
    }
}

struct DrawdownChart: View {
    let data: [DrawdownPoint]
    
    var body: some View {
        Chart(data) { point in
            AreaMark(
                x: .value("Date", point.date),
                y: .value("Drawdown", point.value)
            )
            .foregroundStyle(
                LinearGradient(
                    colors: [
                        Color.red.opacity(0.5),
                        Color.red.opacity(0.1)
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
            
            LineMark(
                x: .value("Date", point.date),
                y: .value("Drawdown", point.value)
            )
            .foregroundStyle(Color.red)
            .lineStyle(StrokeStyle(lineWidth: 1))
        }
        .chartYAxis {
            AxisMarks(position: .leading)
        }
        .chartYScale(domain: .automatic(includesZero: true))
    }
}

struct ReturnsDistributionChart: View {
    let data: [ReturnBin]
    
    var body: some View {
        Chart(data) { bin in
            BarMark(
                x: .value("Return", bin.range),
                y: .value("Frequency", bin.count)
            )
            .foregroundStyle(
                bin.range > 0 ? Color.green : Color.red
            )
        }
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: 10))
        }
    }
}

struct TradesChart: View {
    let trades: [BacktestTrade]
    
    var body: some View {
        Chart(trades.prefix(50)) { trade in
            RectangleMark(
                x: .value("Trade", trade.id),
                y: .value("P&L", trade.pnl)
            )
            .foregroundStyle(trade.pnl > 0 ? Color.green : Color.red)
            .opacity(0.8)
        }
        .chartYAxis {
            AxisMarks(position: .leading)
        }
        .chartXAxis(.hidden)
    }
}

// MARK: - Trade Analysis Section
struct TradeAnalysisSection: View {
    let trades: [BacktestTrade]
    @State private var showAllTrades = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Trade Analysis")
                    .font(.headline)
                
                Spacer()
                
                Text("\(trades.count) trades")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            // Summary Stats
            HStack(spacing: 16) {
                TradeStat(label: "Avg Win", value: "$450", color: .green)
                TradeStat(label: "Avg Loss", value: "$180", color: .red)
                TradeStat(label: "Best Trade", value: "$2,450", color: .green)
                TradeStat(label: "Worst Trade", value: "-$890", color: .red)
            }
            
            // Recent Trades List
            VStack(spacing: 8) {
                ForEach(trades.prefix(showAllTrades ? trades.count : 3)) { trade in
                    TradeRow(trade: trade)
                }
            }
            
            if trades.count > 3 {
                Button(action: { showAllTrades.toggle() }) {
                    Text(showAllTrades ? "Show Less" : "Show All Trades")
                        .font(.caption)
                        .foregroundColor(.blue)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct TradeStat: View {
    let label: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(value)
                .font(.caption)
                .fontWeight(.bold)
                .foregroundColor(color)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct TradeRow: View {
    let trade: BacktestTrade
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(trade.symbol)
                    .font(.subheadline)
                    .fontWeight(.medium)
                Text("\(trade.entryDate, formatter: dateFormatter)")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Text(trade.side == .long ? "LONG" : "SHORT")
                .font(.caption2)
                .fontWeight(.semibold)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(trade.side == .long ? Color.green.opacity(0.2) : Color.red.opacity(0.2))
                .foregroundColor(trade.side == .long ? .green : .red)
                .cornerRadius(4)
            
            Text(formatCurrency(trade.pnl))
                .font(.subheadline)
                .fontWeight(.bold)
                .foregroundColor(trade.pnl > 0 ? .green : .red)
                .frame(width: 80, alignment: .trailing)
        }
        .padding(8)
        .background(Color(.systemGray5))
        .cornerRadius(6)
    }
    
    var dateFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        return formatter
    }
    
    func formatCurrency(_ value: Double) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.maximumFractionDigits = 0
        return formatter.string(from: NSNumber(value: value)) ?? "$0"
    }
}

// MARK: - Risk Metrics Section
struct RiskMetricsSection: View {
    let metrics: RiskMetrics
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Risk Metrics")
                .font(.headline)
            
            VStack(spacing: 8) {
                RiskMetricRow(label: "Value at Risk (95%)", value: formatCurrency(metrics.var95))
                RiskMetricRow(label: "Conditional VaR", value: formatCurrency(metrics.cvar))
                RiskMetricRow(label: "Sortino Ratio", value: String(format: "%.2f", metrics.sortinoRatio))
                RiskMetricRow(label: "Calmar Ratio", value: String(format: "%.2f", metrics.calmarRatio))
                RiskMetricRow(label: "Recovery Factor", value: String(format: "%.2f", metrics.recoveryFactor))
                RiskMetricRow(label: "Avg Drawdown Duration", value: "\(metrics.avgDrawdownDuration) days")
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    func formatCurrency(_ value: Double) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.maximumFractionDigits = 0
        return formatter.string(from: NSNumber(value: value)) ?? "$0"
    }
}

struct RiskMetricRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.subheadline)
                .fontWeight(.medium)
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Monthly Returns Heatmap
struct MonthlyReturnsHeatmap: View {
    let returns: [[Double]]
    let months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    let years = ["2023", "2024", "2025"]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Monthly Returns")
                .font(.headline)
            
            VStack(spacing: 2) {
                // Month headers
                HStack(spacing: 2) {
                    Text("")
                        .frame(width: 40, height: 30)
                    
                    ForEach(months, id: \.self) { month in
                        Text(month)
                            .font(.caption2)
                            .frame(maxWidth: .infinity, minHeight: 30)
                    }
                }
                
                // Year rows with return cells
                ForEach(Array(returns.enumerated()), id: \.offset) { yearIndex, yearReturns in
                    HStack(spacing: 2) {
                        Text(years[safe: yearIndex] ?? "")
                            .font(.caption2)
                            .frame(width: 40, height: 30)
                        
                        ForEach(Array(yearReturns.enumerated()), id: \.offset) { monthIndex, returnValue in
                            ReturnCell(value: returnValue)
                        }
                    }
                }
            }
            
            // Legend
            HStack {
                Text("Loss")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                
                HeatmapLegend()
                
                Text("Profit")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            .padding(.top, 8)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct ReturnCell: View {
    let value: Double
    
    var body: some View {
        Text(String(format: "%.1f%%", value * 100))
            .font(.caption2)
            .fontWeight(.medium)
            .foregroundColor(textColor)
            .frame(maxWidth: .infinity, minHeight: 30)
            .background(backgroundColor)
            .cornerRadius(2)
    }
    
    var backgroundColor: Color {
        if value > 0 {
            return Color.green.opacity(min(abs(value) * 5, 0.8))
        } else if value < 0 {
            return Color.red.opacity(min(abs(value) * 5, 0.8))
        } else {
            return Color.gray.opacity(0.2)
        }
    }
    
    var textColor: Color {
        if abs(value) > 0.1 {
            return .white
        } else {
            return .primary
        }
    }
}

struct HeatmapLegend: View {
    var body: some View {
        HStack(spacing: 2) {
            ForEach(0..<10) { i in
                Rectangle()
                    .fill(
                        i < 5 
                            ? Color.red.opacity(Double(5 - i) / 5 * 0.8)
                            : Color.green.opacity(Double(i - 4) / 5 * 0.8)
                    )
                    .frame(width: 20, height: 10)
            }
        }
        .cornerRadius(2)
    }
}

// MARK: - Supporting Views
struct StrategyPickerSheet: View {
    @Environment(\.dismiss) var dismiss
    @Binding var selectedStrategy: String
    
    let strategies = [
        ("Momentum", "Trend following with momentum indicators", "📈"),
        ("Mean Reversion", "Buy dips, sell rallies", "📊"),
        ("Arbitrage", "Cross-exchange opportunities", "🔄"),
        ("ML Prediction", "Neural network signals", "🤖"),
        ("MACD Cross", "MACD signal crossovers", "📉"),
        ("RSI Oversold", "Buy when RSI < 30", "📊"),
        ("Bollinger Bands", "Trade band breakouts", "📈"),
        ("Custom", "Your custom strategy", "⚙️")
    ]
    
    var body: some View {
        NavigationView {
            List(strategies, id: \.0) { strategy in
                Button(action: {
                    selectedStrategy = strategy.0
                    dismiss()
                }) {
                    HStack {
                        Text(strategy.2)
                            .font(.title2)
                        
                        VStack(alignment: .leading, spacing: 4) {
                            Text(strategy.0)
                                .font(.headline)
                                .foregroundColor(.primary)
                            Text(strategy.1)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        if selectedStrategy == strategy.0 {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.blue)
                        }
                    }
                    .padding(.vertical, 8)
                }
            }
            .navigationTitle("Select Strategy")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

struct DateRangePickerSheet: View {
    @Environment(\.dismiss) var dismiss
    @ObservedObject var manager: BacktestManager
    
    var body: some View {
        NavigationView {
            Form {
                Section("Quick Ranges") {
                    Button("Last Week") {
                        manager.setDateRange(.lastWeek)
                        dismiss()
                    }
                    Button("Last Month") {
                        manager.setDateRange(.lastMonth)
                        dismiss()
                    }
                    Button("Last 3 Months") {
                        manager.setDateRange(.last3Months)
                        dismiss()
                    }
                    Button("Last Year") {
                        manager.setDateRange(.lastYear)
                        dismiss()
                    }
                }
                
                Section("Custom Range") {
                    DatePicker("Start Date", selection: $manager.startDate, displayedComponents: .date)
                    DatePicker("End Date", selection: $manager.endDate, displayedComponents: .date)
                }
            }
            .navigationTitle("Select Date Range")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Apply") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Models
class BacktestManager: ObservableObject {
    @Published var startDate = Date().addingTimeInterval(-30 * 24 * 60 * 60)
    @Published var endDate = Date()
    @Published var initialCapital: Double = 100000
    @Published var positionSize: Double = 0.1
    @Published var stopLoss: Double = 0.02
    @Published var takeProfit: Double = 0.05
    @Published var includeSlippage = true
    @Published var includeFees = true
    @Published var feeRate: Double = 0.001
    
    @Published var hasResults = false
    @Published var results = BacktestResults.mock
    @Published var trades: [BacktestTrade] = BacktestTrade.mockTrades
    @Published var riskMetrics = RiskMetrics.mock
    @Published var monthlyReturns: [[Double]] = MonthlyReturns.mock
    
    enum DateRange {
        case lastWeek, lastMonth, last3Months, lastYear
    }
    
    func setDateRange(_ range: DateRange) {
        endDate = Date()
        switch range {
        case .lastWeek:
            startDate = Date().addingTimeInterval(-7 * 24 * 60 * 60)
        case .lastMonth:
            startDate = Date().addingTimeInterval(-30 * 24 * 60 * 60)
        case .last3Months:
            startDate = Date().addingTimeInterval(-90 * 24 * 60 * 60)
        case .lastYear:
            startDate = Date().addingTimeInterval(-365 * 24 * 60 * 60)
        }
    }
    
    func runBacktest(strategy: String) async {
        // Simulate backtest execution
        try? await Task.sleep(nanoseconds: 2_000_000_000)
        
        await MainActor.run {
            hasResults = true
            // Update with actual backtest results
        }
    }
}

struct BacktestResults {
    let totalReturn: Double
    let annualReturn: Double
    let winRate: Double
    let profitFactor: Double
    let sharpeRatio: Double
    let maxDrawdown: Double
    let finalCapital: Double
    let isProfitable: Bool
    let equityCurve: [EquityPoint]
    let drawdownCurve: [DrawdownPoint]
    let returnsDistribution: [ReturnBin]
    let trades: [BacktestTrade]
    
    static let mock = BacktestResults(
        totalReturn: 0.254,
        annualReturn: 0.312,
        winRate: 0.685,
        profitFactor: 2.15,
        sharpeRatio: 1.85,
        maxDrawdown: -0.125,
        finalCapital: 125400,
        isProfitable: true,
        equityCurve: EquityPoint.mockData,
        drawdownCurve: DrawdownPoint.mockData,
        returnsDistribution: ReturnBin.mockData,
        trades: BacktestTrade.mockTrades
    )
}

struct EquityPoint: Identifiable {
    let id = UUID()
    let date: Date
    let value: Double
    
    static let mockData: [EquityPoint] = {
        let baseValue = 100000.0
        return (0..<100).map { i in
            let progress = Double(i) / 100
            let value = baseValue * (1 + progress * 0.25 + sin(progress * .pi * 4) * 0.05)
            return EquityPoint(
                date: Date().addingTimeInterval(-86400 * Double(100 - i)),
                value: value
            )
        }
    }()
}

struct DrawdownPoint: Identifiable {
    let id = UUID()
    let date: Date
    let value: Double
    
    static let mockData: [DrawdownPoint] = {
        (0..<100).map { i in
            let progress = Double(i) / 100
            let value = -abs(sin(progress * .pi * 3) * 0.125)
            return DrawdownPoint(
                date: Date().addingTimeInterval(-86400 * Double(100 - i)),
                value: value
            )
        }
    }()
}

struct ReturnBin: Identifiable {
    let id = UUID()
    let range: String
    let count: Int
    
    static let mockData = [
        ReturnBin(range: "-5%", count: 2),
        ReturnBin(range: "-4%", count: 3),
        ReturnBin(range: "-3%", count: 5),
        ReturnBin(range: "-2%", count: 8),
        ReturnBin(range: "-1%", count: 12),
        ReturnBin(range: "0%", count: 15),
        ReturnBin(range: "1%", count: 18),
        ReturnBin(range: "2%", count: 14),
        ReturnBin(range: "3%", count: 10),
        ReturnBin(range: "4%", count: 6),
        ReturnBin(range: "5%", count: 4)
    ]
}

struct BacktestTrade: Identifiable {
    let id = UUID()
    let symbol: String
    let side: TradeSide
    let entryDate: Date
    let exitDate: Date
    let entryPrice: Double
    let exitPrice: Double
    let quantity: Double
    let pnl: Double
    
    static let mockTrades: [BacktestTrade] = {
        let symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        return (0..<20).map { i in
            let symbol = symbols[i % symbols.count]
            let side = i % 3 == 0 ? TradeSide.short : TradeSide.long
            let entryPrice = Double.random(in: 40000...60000)
            let exitPrice = entryPrice * (1 + Double.random(in: -0.05...0.08))
            let quantity = Double.random(in: 0.01...0.5)
            let pnl = (exitPrice - entryPrice) * quantity * (side == .long ? 1 : -1)
            
            return BacktestTrade(
                symbol: symbol,
                side: side,
                entryDate: Date().addingTimeInterval(-86400 * Double(20 - i)),
                exitDate: Date().addingTimeInterval(-86400 * Double(19 - i)),
                entryPrice: entryPrice,
                exitPrice: exitPrice,
                quantity: quantity,
                pnl: pnl
            )
        }
    }()
}

struct RiskMetrics {
    let var95: Double
    let cvar: Double
    let sortinoRatio: Double
    let calmarRatio: Double
    let recoveryFactor: Double
    let avgDrawdownDuration: Int
    
    static let mock = RiskMetrics(
        var95: -2450,
        cvar: -3200,
        sortinoRatio: 2.15,
        calmarRatio: 2.50,
        recoveryFactor: 3.85,
        avgDrawdownDuration: 12
    )
}

struct MonthlyReturns {
    static let mock: [[Double]] = [
        [0.023, -0.015, 0.041, 0.032, -0.008, 0.055, 0.021, -0.032, 0.045, 0.018, 0.062, 0.035],
        [0.015, 0.028, -0.022, 0.048, 0.012, -0.018, 0.038, 0.025, -0.012, 0.052, 0.031, 0.042],
        [0.032, 0.018, 0.025, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
    ]
}

// MARK: - Extension
extension Collection {
    subscript(safe index: Index) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}