import SwiftUI
import Charts

struct DashboardView: View {
    @EnvironmentObject var marketDataManager: MarketDataManager
    @EnvironmentObject var portfolioManager: PortfolioManager
    @EnvironmentObject var tradingManager: TradingManager
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Portfolio Summary
                    PortfolioSummaryCard()
                    
                    // Market Overview
                    MarketOverviewSection()
                    
                    // Quick Actions
                    QuickActionsCard()
                    
                    // Recent Activity
                    RecentActivityCard()
                    
                    // Performance Chart
                    PerformanceChartCard()
                }
                .padding()
            }
            .navigationTitle("Dashboard")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        // Refresh data
                    }) {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
        }
    }
}

// MARK: - Portfolio Summary Card
struct PortfolioSummaryCard: View {
    @EnvironmentObject var portfolioManager: PortfolioManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Portfolio Value")
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text(formatCurrency(portfolioManager.totalValue))
                .font(.system(size: 36, weight: .bold, design: .rounded))
                .foregroundColor(.green)
            
            HStack {
                Label(
                    formatCurrency(portfolioManager.dailyPnL),
                    systemImage: portfolioManager.dailyPnL >= 0 ? "arrow.up.right" : "arrow.down.right"
                )
                .foregroundColor(portfolioManager.dailyPnL >= 0 ? .green : .red)
                
                Text("(\(String(format: "%+.2f%%", portfolioManager.dailyPnLPercent)))")
                    .foregroundColor(portfolioManager.dailyPnL >= 0 ? .green : .red)
                
                Text("Today")
                    .foregroundColor(.secondary)
            }
            .font(.subheadline)
            
            Divider()
            
            HStack {
                VStack(alignment: .leading) {
                    Text("Cash")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(formatCurrency(portfolioManager.cash))
                        .font(.headline)
                }
                
                Spacer()
                
                VStack(alignment: .trailing) {
                    Text("Invested")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(formatCurrency(portfolioManager.invested))
                        .font(.headline)
                }
            }
        }
        .padding()
        .background(
            LinearGradient(
                gradient: Gradient(colors: [
                    Color.green.opacity(0.1),
                    Color.green.opacity(0.05)
                ]),
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .cornerRadius(16)
    }
    
    func formatCurrency(_ value: Double) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.maximumFractionDigits = 2
        return formatter.string(from: NSNumber(value: value)) ?? "$0.00"
    }
}

// MARK: - Market Overview Section
struct MarketOverviewSection: View {
    @EnvironmentObject var marketDataManager: MarketDataManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Market Overview")
                .font(.headline)
            
            ForEach(Array(marketDataManager.prices.values)) { crypto in
                MarketRowView(crypto: crypto)
            }
        }
    }
}

struct MarketRowView: View {
    let crypto: CryptoPrice
    @State private var sparklineData: [Double] = []
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(crypto.symbol)
                    .font(.headline)
                Text("Vol: $\(String(format: "%.1fB", crypto.volume))")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            // Mini chart
            Chart {
                ForEach(Array(sparklineData.enumerated()), id: \.offset) { index, value in
                    LineMark(
                        x: .value("Time", index),
                        y: .value("Price", value)
                    )
                    .foregroundStyle(crypto.changeColor)
                }
            }
            .frame(width: 80, height: 40)
            .chartXAxis(.hidden)
            .chartYAxis(.hidden)
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 4) {
                Text(crypto.formattedPrice)
                    .font(.headline)
                    .fontWeight(.bold)
                Text(crypto.formattedChange)
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(crypto.changeColor)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(crypto.changeColor.opacity(0.2))
                    .cornerRadius(4)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
        .onAppear {
            generateSparklineData()
        }
    }
    
    func generateSparklineData() {
        sparklineData = (0..<10).map { _ in
            crypto.price * Double.random(in: 0.98...1.02)
        }
    }
}

// MARK: - Quick Actions Card
struct QuickActionsCard: View {
    @EnvironmentObject var tradingManager: TradingManager
    @State private var showTradingSheet = false
    @State private var showAIChat = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Quick Actions")
                .font(.headline)
            
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 12) {
                QuickActionButton(
                    title: "Buy",
                    icon: "plus.circle.fill",
                    color: .green,
                    action: { showTradingSheet = true }
                )
                
                QuickActionButton(
                    title: "Sell",
                    icon: "minus.circle.fill",
                    color: .red,
                    action: { showTradingSheet = true }
                )
                
                QuickActionButton(
                    title: "AI Chat",
                    icon: "message.fill",
                    color: .blue,
                    action: { showAIChat = true }
                )
                
                QuickActionButton(
                    title: "Voice",
                    icon: "mic.fill",
                    color: .purple,
                    action: { }
                )
            }
        }
        .sheet(isPresented: $showTradingSheet) {
            TradingSheet()
        }
        .sheet(isPresented: $showAIChat) {
            AIChatView()
        }
    }
}

struct QuickActionButton: View {
    let title: String
    let icon: String
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: icon)
                    .font(.title3)
                Text(title)
                    .fontWeight(.medium)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .foregroundColor(.white)
            .background(color)
            .cornerRadius(12)
        }
    }
}

// MARK: - Recent Activity Card
struct RecentActivityCard: View {
    let activities = [
        ("Trade Executed", "BTC: Buy 0.1 @ $110,768", "checkmark.circle.fill", Color.green),
        ("Alert Triggered", "ETH crossed $3,750", "bell.fill", Color.orange),
        ("Position Closed", "SOL: +5.2% profit", "xmark.circle.fill", Color.blue)
    ]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Recent Activity")
                    .font(.headline)
                Spacer()
                Button("See All") {
                    // Navigate to activity log
                }
                .font(.caption)
            }
            
            ForEach(activities, id: \.0) { activity in
                HStack {
                    Image(systemName: activity.2)
                        .foregroundColor(activity.3)
                        .font(.title3)
                        .frame(width: 30)
                    
                    VStack(alignment: .leading, spacing: 2) {
                        Text(activity.0)
                            .font(.subheadline)
                            .fontWeight(.medium)
                        Text(activity.1)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    Text("2m ago")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 8)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Performance Chart Card
struct PerformanceChartCard: View {
    @State private var selectedTimeframe = "1D"
    let timeframes = ["1H", "1D", "1W", "1M", "1Y"]
    @State private var chartData: [ChartDataPoint] = []
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Performance")
                    .font(.headline)
                
                Spacer()
                
                Picker("Timeframe", selection: $selectedTimeframe) {
                    ForEach(timeframes, id: \.self) { timeframe in
                        Text(timeframe).tag(timeframe)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .frame(width: 200)
            }
            
            Chart(chartData) { dataPoint in
                LineMark(
                    x: .value("Time", dataPoint.date),
                    y: .value("Value", dataPoint.value)
                )
                .foregroundStyle(
                    LinearGradient(
                        gradient: Gradient(colors: [.green, .blue]),
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                
                AreaMark(
                    x: .value("Time", dataPoint.date),
                    y: .value("Value", dataPoint.value)
                )
                .foregroundStyle(
                    LinearGradient(
                        gradient: Gradient(colors: [
                            Color.green.opacity(0.3),
                            Color.green.opacity(0.1)
                        ]),
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
            }
            .frame(height: 200)
            .chartYAxis {
                AxisMarks(position: .leading)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
        .onAppear {
            generateChartData()
        }
        .onChange(of: selectedTimeframe) { _ in
            generateChartData()
        }
    }
    
    func generateChartData() {
        let baseValue = 100000.0
        let points = 50
        chartData = (0..<points).map { i in
            let progress = Double(i) / Double(points)
            let value = baseValue * (1 + sin(progress * .pi * 2) * 0.1 + progress * 0.25)
            let date = Date().addingTimeInterval(-3600 * Double(points - i))
            return ChartDataPoint(date: date, value: value)
        }
    }
}

struct ChartDataPoint: Identifiable {
    let id = UUID()
    let date: Date
    let value: Double
}

// MARK: - Trading Sheet
struct TradingSheet: View {
    @Environment(\.dismiss) var dismiss
    @State private var selectedSymbol = "BTC-USD"
    @State private var orderType = "Market"
    @State private var quantity = ""
    @State private var price = ""
    @State private var side = "Buy"
    
    let symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOGE-USD"]
    let orderTypes = ["Market", "Limit", "Stop"]
    let sides = ["Buy", "Sell"]
    
    var body: some View {
        NavigationView {
            Form {
                Section("Order Details") {
                    Picker("Symbol", selection: $selectedSymbol) {
                        ForEach(symbols, id: \.self) { symbol in
                            Text(symbol).tag(symbol)
                        }
                    }
                    
                    Picker("Side", selection: $side) {
                        ForEach(sides, id: \.self) { side in
                            Text(side).tag(side)
                        }
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    
                    Picker("Order Type", selection: $orderType) {
                        ForEach(orderTypes, id: \.self) { type in
                            Text(type).tag(type)
                        }
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    
                    TextField("Quantity", text: $quantity)
                        .keyboardType(.decimalPad)
                    
                    if orderType != "Market" {
                        TextField("Price", text: $price)
                            .keyboardType(.decimalPad)
                    }
                }
                
                Section {
                    Button(action: {
                        // Place order
                        dismiss()
                    }) {
                        Text("Place Order")
                            .frame(maxWidth: .infinity)
                            .fontWeight(.bold)
                    }
                    .foregroundColor(.white)
                    .listRowBackground(side == "Buy" ? Color.green : Color.red)
                }
            }
            .navigationTitle("New Order")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Preview
struct DashboardView_Previews: PreviewProvider {
    static var previews: some View {
        DashboardView()
            .environmentObject(MarketDataManager())
            .environmentObject(PortfolioManager())
            .environmentObject(TradingManager())
            .preferredColorScheme(.dark)
    }
}