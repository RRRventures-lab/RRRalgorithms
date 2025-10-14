import SwiftUI
import Charts

struct PortfolioView: View {
    @EnvironmentObject var portfolioManager: PortfolioManager
    @State private var selectedSegment = 0
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Segment Control
                    Picker("View", selection: $selectedSegment) {
                        Text("Positions").tag(0)
                        Text("Performance").tag(1)
                        Text("Analytics").tag(2)
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    .padding(.horizontal)
                    
                    switch selectedSegment {
                    case 0:
                        PositionsView()
                    case 1:
                        PerformanceView()
                    case 2:
                        AnalyticsView()
                    default:
                        PositionsView()
                    }
                }
            }
            .navigationTitle("Portfolio")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}

// MARK: - Positions View
struct PositionsView: View {
    @EnvironmentObject var portfolioManager: PortfolioManager
    
    var body: some View {
        VStack(spacing: 16) {
            ForEach(portfolioManager.positions) { position in
                PositionCard(position: position)
            }
            
            if portfolioManager.positions.isEmpty {
                EmptyPositionsView()
            }
        }
        .padding()
    }
}

struct PositionCard: View {
    let position: Position
    @State private var showDetails = false
    
    var body: some View {
        VStack(spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(position.symbol)
                        .font(.headline)
                    Text("\(String(format: "%.4f", position.quantity)) units")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text(formatCurrency(position.currentPrice * position.quantity))
                        .font(.headline)
                    HStack(spacing: 4) {
                        Image(systemName: position.pnl >= 0 ? "arrow.up.right" : "arrow.down.right")
                            .font(.caption)
                        Text(formatCurrency(position.pnl))
                        Text("(\(String(format: "%+.2f%%", position.pnlPercent)))")
                    }
                    .font(.caption)
                    .foregroundColor(position.pnlColor)
                }
            }
            
            if showDetails {
                Divider()
                
                VStack(spacing: 8) {
                    DetailRow(label: "Entry Price", value: formatCurrency(position.entryPrice))
                    DetailRow(label: "Current Price", value: formatCurrency(position.currentPrice))
                    DetailRow(label: "Position Value", value: formatCurrency(position.currentPrice * position.quantity))
                    DetailRow(label: "Side", value: position.side == .long ? "Long" : "Short")
                }
                
                HStack(spacing: 12) {
                    Button(action: {}) {
                        Label("Add", systemImage: "plus.circle")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(BorderedButtonStyle())
                    .tint(.green)
                    
                    Button(action: {}) {
                        Label("Close", systemImage: "xmark.circle")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(BorderedButtonStyle())
                    .tint(.red)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
        .onTapGesture {
            withAnimation {
                showDetails.toggle()
            }
        }
    }
    
    func formatCurrency(_ value: Double) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.maximumFractionDigits = 2
        return formatter.string(from: NSNumber(value: value)) ?? "$0.00"
    }
}

struct DetailRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
        }
    }
}

struct EmptyPositionsView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "tray")
                .font(.system(size: 48))
                .foregroundColor(.secondary)
            
            Text("No Open Positions")
                .font(.headline)
            
            Text("Start trading to see your positions here")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            
            Button(action: {}) {
                Label("Start Trading", systemImage: "plus.circle.fill")
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
            }
            .buttonStyle(BorderedProminentButtonStyle())
            .tint(.green)
        }
        .frame(maxWidth: .infinity)
        .padding(40)
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Performance View
struct PerformanceView: View {
    @EnvironmentObject var portfolioManager: PortfolioManager
    @State private var selectedMetric = "Equity"
    let metrics = ["Equity", "P&L", "Win Rate", "Sharpe"]
    
    var body: some View {
        VStack(spacing: 20) {
            // Performance Chart
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Performance")
                        .font(.headline)
                    
                    Spacer()
                    
                    Picker("Metric", selection: $selectedMetric) {
                        ForEach(metrics, id: \.self) { metric in
                            Text(metric).tag(metric)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                }
                
                PerformanceChart(metric: selectedMetric)
                    .frame(height: 250)
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(12)
            
            // Key Metrics Grid
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 12) {
                MetricCard(title: "Total Return", value: "+25.43%", color: .green)
                MetricCard(title: "Sharpe Ratio", value: "1.85", color: .blue)
                MetricCard(title: "Win Rate", value: "68.5%", color: .green)
                MetricCard(title: "Max Drawdown", value: "-8.5%", color: .red)
                MetricCard(title: "Avg Win", value: "$450", color: .green)
                MetricCard(title: "Avg Loss", value: "$180", color: .red)
            }
        }
        .padding()
    }
}

struct PerformanceChart: View {
    let metric: String
    @State private var chartData: [ChartDataPoint] = []
    
    var body: some View {
        Chart(chartData) { dataPoint in
            LineMark(
                x: .value("Date", dataPoint.date),
                y: .value("Value", dataPoint.value)
            )
            .foregroundStyle(Color.green)
            .lineStyle(StrokeStyle(lineWidth: 2))
            
            AreaMark(
                x: .value("Date", dataPoint.date),
                y: .value("Value", dataPoint.value)
            )
            .foregroundStyle(
                LinearGradient(
                    colors: [
                        Color.green.opacity(0.3),
                        Color.green.opacity(0.05)
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
        }
        .onAppear {
            generateData()
        }
        .onChange(of: metric) { _ in
            generateData()
        }
    }
    
    func generateData() {
        let days = 30
        let baseValue = metric == "Win Rate" ? 50.0 : 100000.0
        
        chartData = (0..<days).map { i in
            let progress = Double(i) / Double(days)
            let randomFactor = Double.random(in: 0.95...1.05)
            let trend = metric == "P&L" ? progress * 0.25 : progress * 0.1
            let value = baseValue * (1 + trend) * randomFactor
            
            return ChartDataPoint(
                date: Date().addingTimeInterval(-86400 * Double(days - i)),
                value: value
            )
        }
    }
}

struct MetricCard: View {
    let title: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            Text(value)
                .font(.title3)
                .fontWeight(.bold)
                .foregroundColor(color)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

// MARK: - Analytics View
struct AnalyticsView: View {
    var body: some View {
        VStack(spacing: 20) {
            // Risk Metrics
            RiskMetricsCard()
            
            // Asset Allocation
            AssetAllocationCard()
            
            // Correlation Matrix
            CorrelationMatrixCard()
        }
        .padding()
    }
}

struct RiskMetricsCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Risk Metrics")
                .font(.headline)
            
            VStack(spacing: 12) {
                RiskMetricRow(label: "Value at Risk (95%)", value: "$2,450", status: .safe)
                RiskMetricRow(label: "Beta", value: "1.12", status: .moderate)
                RiskMetricRow(label: "Correlation to BTC", value: "0.85", status: .high)
                RiskMetricRow(label: "Volatility (30d)", value: "42.5%", status: .moderate)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct RiskMetricRow: View {
    let label: String
    let value: String
    let status: RiskStatus
    
    enum RiskStatus {
        case safe, moderate, high
        
        var color: Color {
            switch self {
            case .safe: return .green
            case .moderate: return .orange
            case .high: return .red
            }
        }
    }
    
    var body: some View {
        HStack {
            Circle()
                .fill(status.color)
                .frame(width: 8, height: 8)
            
            Text(label)
                .font(.subheadline)
            
            Spacer()
            
            Text(value)
                .font(.subheadline)
                .fontWeight(.medium)
        }
    }
}

struct AssetAllocationCard: View {
    @State private var allocations = [
        ("BTC", 0.45, Color.orange),
        ("ETH", 0.30, Color.blue),
        ("SOL", 0.15, Color.purple),
        ("Others", 0.10, Color.gray)
    ]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Asset Allocation")
                .font(.headline)
            
            // Pie Chart Placeholder
            ZStack {
                ForEach(Array(allocations.enumerated()), id: \.offset) { index, allocation in
                    PieSlice(
                        startAngle: startAngle(for: index),
                        endAngle: endAngle(for: index),
                        color: allocation.2
                    )
                }
            }
            .frame(height: 200)
            
            // Legend
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 8) {
                ForEach(allocations, id: \.0) { allocation in
                    HStack(spacing: 8) {
                        Circle()
                            .fill(allocation.2)
                            .frame(width: 12, height: 12)
                        Text(allocation.0)
                            .font(.caption)
                        Spacer()
                        Text("\(Int(allocation.1 * 100))%")
                            .font(.caption)
                            .fontWeight(.medium)
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    func startAngle(for index: Int) -> Angle {
        let angles = allocations.prefix(index).map { $0.1 * 360 }
        return Angle(degrees: angles.reduce(0, +) - 90)
    }
    
    func endAngle(for index: Int) -> Angle {
        let angles = allocations.prefix(index + 1).map { $0.1 * 360 }
        return Angle(degrees: angles.reduce(0, +) - 90)
    }
}

struct PieSlice: View {
    let startAngle: Angle
    let endAngle: Angle
    let color: Color
    
    var body: some View {
        GeometryReader { geometry in
            Path { path in
                let center = CGPoint(
                    x: geometry.size.width / 2,
                    y: geometry.size.height / 2
                )
                let radius = min(geometry.size.width, geometry.size.height) / 2
                
                path.move(to: center)
                path.addArc(
                    center: center,
                    radius: radius,
                    startAngle: startAngle,
                    endAngle: endAngle,
                    clockwise: false
                )
                path.closeSubpath()
            }
            .fill(color)
        }
    }
}

struct CorrelationMatrixCard: View {
    let correlations = [
        ["BTC", "1.00", "0.85", "0.72"],
        ["ETH", "0.85", "1.00", "0.68"],
        ["SOL", "0.72", "0.68", "1.00"]
    ]
    let headers = ["", "BTC", "ETH", "SOL"]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Correlation Matrix")
                .font(.headline)
            
            VStack(spacing: 0) {
                // Headers
                HStack(spacing: 0) {
                    ForEach(headers, id: \.self) { header in
                        Text(header)
                            .font(.caption)
                            .fontWeight(.medium)
                            .frame(maxWidth: .infinity)
                            .padding(8)
                    }
                }
                .background(Color(.systemGray5))
                
                // Data rows
                ForEach(correlations, id: \.0) { row in
                    HStack(spacing: 0) {
                        ForEach(Array(row.enumerated()), id: \.offset) { index, value in
                            Text(value)
                                .font(.caption)
                                .fontWeight(index == 0 ? .medium : .regular)
                                .frame(maxWidth: .infinity)
                                .padding(8)
                                .background(
                                    index == 0 
                                        ? Color(.systemGray5) 
                                        : Color.blue.opacity(Double(value) ?? 0.0 * 0.3)
                                )
                        }
                    }
                }
            }
            .cornerRadius(8)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}