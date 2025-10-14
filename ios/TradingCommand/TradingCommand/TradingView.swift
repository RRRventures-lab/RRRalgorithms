import SwiftUI
import Charts

struct TradingView: View {
    @EnvironmentObject var marketDataManager: MarketDataManager
    @State private var selectedSymbol = "BTC-USD"
    @State private var selectedTimeframe = "1H"
    @State private var showOrderSheet = false
    
    let symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOGE-USD"]
    let timeframes = ["1M", "5M", "15M", "1H", "4H", "1D"]
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Symbol Selector
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        ForEach(symbols, id: \.self) { symbol in
                            SymbolChip(
                                symbol: symbol,
                                isSelected: symbol == selectedSymbol,
                                price: marketDataManager.prices[symbol]
                            ) {
                                selectedSymbol = symbol
                            }
                        }
                    }
                    .padding()
                }
                
                // Main Chart
                TradingChartView(symbol: selectedSymbol, timeframe: selectedTimeframe)
                    .frame(height: 400)
                    .padding()
                
                // Timeframe Selector
                Picker("Timeframe", selection: $selectedTimeframe) {
                    ForEach(timeframes, id: \.self) { timeframe in
                        Text(timeframe).tag(timeframe)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding(.horizontal)
                
                // Order Book & Info
                HStack(spacing: 16) {
                    OrderBookMini(symbol: selectedSymbol)
                    MarketInfoCard(symbol: selectedSymbol)
                }
                .padding()
                
                Spacer()
                
                // Trading Buttons
                HStack(spacing: 12) {
                    Button(action: { showOrderSheet = true }) {
                        Label("Buy", systemImage: "plus.circle.fill")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .foregroundColor(.white)
                            .background(Color.green)
                            .cornerRadius(12)
                    }
                    
                    Button(action: { showOrderSheet = true }) {
                        Label("Sell", systemImage: "minus.circle.fill")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .foregroundColor(.white)
                            .background(Color.red)
                            .cornerRadius(12)
                    }
                }
                .padding()
            }
            .navigationTitle("Trading")
            .navigationBarTitleDisplayMode(.inline)
            .sheet(isPresented: $showOrderSheet) {
                OrderSheet(symbol: selectedSymbol)
            }
        }
    }
}

// MARK: - Symbol Chip
struct SymbolChip: View {
    let symbol: String
    let isSelected: Bool
    let price: CryptoPrice?
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Text(symbol.replacingOccurrences(of: "-USD", with: ""))
                    .font(.headline)
                
                if let price = price {
                    Text(price.formattedPrice)
                        .font(.caption)
                        .fontWeight(.medium)
                    
                    Text(price.formattedChange)
                        .font(.caption2)
                        .foregroundColor(price.changeColor)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(
                isSelected 
                    ? Color.blue 
                    : Color(.systemGray6)
            )
            .foregroundColor(isSelected ? .white : .primary)
            .cornerRadius(12)
        }
    }
}

// MARK: - Trading Chart
struct TradingChartView: View {
    let symbol: String
    let timeframe: String
    @State private var candleData: [CandleData] = []
    @State private var showIndicators = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Chart Header
            HStack {
                VStack(alignment: .leading) {
                    Text(symbol)
                        .font(.headline)
                    Text("Last update: \(Date(), style: .time)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Button(action: { showIndicators.toggle() }) {
                    Image(systemName: "chart.xyaxis.line")
                        .foregroundColor(showIndicators ? .blue : .gray)
                }
            }
            .padding(.bottom, 12)
            
            // Candlestick Chart
            Chart(candleData) { candle in
                RectangleMark(
                    x: .value("Time", candle.date),
                    yStart: .value("Low", candle.low),
                    yEnd: .value("High", candle.high),
                    width: 1
                )
                .foregroundStyle(Color.gray)
                
                RectangleMark(
                    x: .value("Time", candle.date),
                    yStart: .value("Open", candle.open),
                    yEnd: .value("Close", candle.close),
                    width: 8
                )
                .foregroundStyle(candle.close > candle.open ? Color.green : Color.red)
            }
            .chartYAxis {
                AxisMarks(position: .trailing)
            }
            .chartXAxis {
                AxisMarks(values: .automatic(desiredCount: 6))
            }
            
            if showIndicators {
                // Volume Chart
                Chart(candleData) { candle in
                    BarMark(
                        x: .value("Time", candle.date),
                        y: .value("Volume", candle.volume)
                    )
                    .foregroundStyle(Color.gray.opacity(0.5))
                }
                .frame(height: 60)
                .chartXAxis(.hidden)
                .chartYAxis {
                    AxisMarks(position: .trailing)
                }
                .padding(.top, 12)
            }
        }
        .onAppear {
            generateCandleData()
        }
        .onChange(of: timeframe) { _ in
            generateCandleData()
        }
        .onChange(of: symbol) { _ in
            generateCandleData()
        }
    }
    
    func generateCandleData() {
        let basePrice = 110000.0
        let candles = 50
        
        candleData = (0..<candles).map { i in
            let time = Date().addingTimeInterval(-3600 * Double(candles - i))
            let open = basePrice * (1 + Double.random(in: -0.02...0.02))
            let close = open * (1 + Double.random(in: -0.015...0.015))
            let high = max(open, close) * (1 + Double.random(in: 0...0.01))
            let low = min(open, close) * (1 - Double.random(in: 0...0.01))
            let volume = Double.random(in: 1000...10000)
            
            return CandleData(
                date: time,
                open: open,
                high: high,
                low: low,
                close: close,
                volume: volume
            )
        }
    }
}

struct CandleData: Identifiable {
    let id = UUID()
    let date: Date
    let open: Double
    let high: Double
    let low: Double
    let close: Double
    let volume: Double
}

// MARK: - Order Book Mini
struct OrderBookMini: View {
    let symbol: String
    @State private var bids: [(price: Double, amount: Double)] = []
    @State private var asks: [(price: Double, amount: Double)] = []
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Order Book")
                .font(.caption)
                .fontWeight(.medium)
            
            VStack(spacing: 2) {
                // Asks
                ForEach(asks.prefix(3), id: \.price) { ask in
                    HStack {
                        Text(String(format: "%.2f", ask.amount))
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text(String(format: "%.2f", ask.price))
                            .font(.caption2)
                            .foregroundColor(.red)
                    }
                    .background(
                        GeometryReader { geometry in
                            Rectangle()
                                .fill(Color.red.opacity(0.1))
                                .frame(width: geometry.size.width * (ask.amount / 10))
                                .frame(maxWidth: .infinity, alignment: .trailing)
                        }
                    )
                }
                
                Divider()
                
                // Bids
                ForEach(bids.prefix(3), id: \.price) { bid in
                    HStack {
                        Text(String(format: "%.2f", bid.price))
                            .font(.caption2)
                            .foregroundColor(.green)
                        Spacer()
                        Text(String(format: "%.2f", bid.amount))
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    .background(
                        GeometryReader { geometry in
                            Rectangle()
                                .fill(Color.green.opacity(0.1))
                                .frame(width: geometry.size.width * (bid.amount / 10))
                        }
                    )
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
        .onAppear {
            generateOrderBook()
        }
    }
    
    func generateOrderBook() {
        let basePrice = 110000.0
        
        bids = (0..<5).map { i in
            (
                price: basePrice - Double(i + 1) * 10,
                amount: Double.random(in: 0.1...5.0)
            )
        }
        
        asks = (0..<5).map { i in
            (
                price: basePrice + Double(i + 1) * 10,
                amount: Double.random(in: 0.1...5.0)
            )
        }.reversed()
    }
}

// MARK: - Market Info Card
struct MarketInfoCard: View {
    let symbol: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Market Info")
                .font(.caption)
                .fontWeight(.medium)
            
            VStack(alignment: .leading, spacing: 6) {
                InfoRow(label: "24h High", value: "$113,000")
                InfoRow(label: "24h Low", value: "$108,500")
                InfoRow(label: "24h Volume", value: "$28.5B")
                InfoRow(label: "Market Cap", value: "$2.1T")
                InfoRow(label: "Circulating", value: "19.5M")
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct InfoRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.caption2)
                .fontWeight(.medium)
        }
    }
}

// MARK: - Order Sheet
struct OrderSheet: View {
    @Environment(\.dismiss) var dismiss
    let symbol: String
    
    @State private var orderType = "Market"
    @State private var side = "Buy"
    @State private var quantity = ""
    @State private var price = ""
    @State private var stopPrice = ""
    @State private var takeProfit = ""
    @State private var stopLoss = ""
    
    let orderTypes = ["Market", "Limit", "Stop-Limit"]
    let sides = ["Buy", "Sell"]
    
    var body: some View {
        NavigationView {
            Form {
                Section("Order Type") {
                    Picker("Type", selection: $orderType) {
                        ForEach(orderTypes, id: \.self) { type in
                            Text(type).tag(type)
                        }
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    
                    Picker("Side", selection: $side) {
                        ForEach(sides, id: \.self) { side in
                            Text(side).tag(side)
                        }
                    }
                    .pickerStyle(SegmentedPickerStyle())
                }
                
                Section("Order Details") {
                    HStack {
                        Text("Symbol")
                        Spacer()
                        Text(symbol)
                            .fontWeight(.medium)
                    }
                    
                    TextField("Quantity", text: $quantity)
                        .keyboardType(.decimalPad)
                    
                    if orderType != "Market" {
                        TextField("Limit Price", text: $price)
                            .keyboardType(.decimalPad)
                    }
                    
                    if orderType == "Stop-Limit" {
                        TextField("Stop Price", text: $stopPrice)
                            .keyboardType(.decimalPad)
                    }
                }
                
                Section("Risk Management (Optional)") {
                    TextField("Take Profit", text: $takeProfit)
                        .keyboardType(.decimalPad)
                    
                    TextField("Stop Loss", text: $stopLoss)
                        .keyboardType(.decimalPad)
                }
                
                Section("Order Summary") {
                    HStack {
                        Text("Est. Total")
                        Spacer()
                        Text("$0.00")
                            .fontWeight(.bold)
                    }
                    
                    HStack {
                        Text("Fee")
                        Spacer()
                        Text("$0.00")
                            .foregroundColor(.secondary)
                    }
                }
                
                Section {
                    Button(action: placeOrder) {
                        Text("Place \(side) Order")
                            .frame(maxWidth: .infinity)
                            .fontWeight(.bold)
                    }
                    .foregroundColor(.white)
                    .listRowBackground(side == "Buy" ? Color.green : Color.red)
                }
            }
            .navigationTitle("Place Order")
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
    
    func placeOrder() {
        // Place order logic
        dismiss()
    }
}

// MARK: - Preview
struct TradingView_Previews: PreviewProvider {
    static var previews: some View {
        TradingView()
            .environmentObject(MarketDataManager())
            .preferredColorScheme(.dark)
    }
}