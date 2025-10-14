import SwiftUI
import Charts
import Combine

@main
struct TradingCommandApp: App {
    @StateObject private var marketDataManager = MarketDataManager()
    @StateObject private var portfolioManager = PortfolioManager()
    @StateObject private var tradingManager = TradingManager()
    @StateObject private var voiceCommandManager = VoiceCommandManager()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(marketDataManager)
                .environmentObject(portfolioManager)
                .environmentObject(tradingManager)
                .environmentObject(voiceCommandManager)
                .preferredColorScheme(.dark)
        }
    }
}

// MARK: - Market Data Manager
class MarketDataManager: ObservableObject {
    @Published var prices: [String: CryptoPrice] = [:]
    @Published var isConnected = false
    
    init() {
        setupMockData()
    }
    
    func setupMockData() {
        prices = [
            "BTC-USD": CryptoPrice(symbol: "BTC-USD", price: 110768.89, change24h: -1.95, volume: 28.5),
            "ETH-USD": CryptoPrice(symbol: "ETH-USD", price: 3750.60, change24h: -2.25, volume: 15.6),
            "SOL-USD": CryptoPrice(symbol: "SOL-USD", price: 177.82, change24h: -5.83, volume: 2.1)
        ]
    }
}

// MARK: - Portfolio Manager
class PortfolioManager: ObservableObject {
    @Published var totalValue: Double = 125430.50
    @Published var dailyPnL: Double = 2340.50
    @Published var dailyPnLPercent: Double = 1.90
    @Published var positions: [Position] = []
    @Published var cash: Double = 45230.25
    @Published var invested: Double = 80200.25
    
    init() {
        setupMockPositions()
    }
    
    func setupMockPositions() {
        positions = [
            Position(id: "1", symbol: "BTC-USD", quantity: 0.5, entryPrice: 108000, currentPrice: 110768.89, side: .long),
            Position(id: "2", symbol: "ETH-USD", quantity: 10, entryPrice: 3650, currentPrice: 3750.60, side: .long)
        ]
    }
}

// MARK: - Trading Manager
class TradingManager: ObservableObject {
    @Published var isTradingEnabled = false
    @Published var systemStatus: SystemStatus = .ready
    @Published var riskLevel: Double = 30
    @Published var maxPositionSize: Double = 1000
    
    func toggleTrading() {
        isTradingEnabled.toggle()
        systemStatus = isTradingEnabled ? .running : .paused
    }
    
    func emergencyStop() {
        isTradingEnabled = false
        systemStatus = .stopped
        // Close all positions
    }
}

// MARK: - Voice Command Manager
class VoiceCommandManager: ObservableObject {
    @Published var isListening = false
    @Published var lastCommand = ""
    @Published var commandHistory: [VoiceCommand] = []
    
    func startListening() {
        isListening = true
    }
    
    func stopListening() {
        isListening = false
    }
    
    func processCommand(_ command: String) {
        lastCommand = command
        commandHistory.append(VoiceCommand(text: command, timestamp: Date()))
    }
}

// MARK: - Data Models
struct CryptoPrice: Identifiable {
    let id = UUID()
    let symbol: String
    let price: Double
    let change24h: Double
    let volume: Double
    
    var changeColor: Color {
        change24h >= 0 ? .green : .red
    }
    
    var formattedPrice: String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.maximumFractionDigits = price < 1 ? 4 : 2
        return formatter.string(from: NSNumber(value: price)) ?? "$0.00"
    }
    
    var formattedChange: String {
        String(format: "%+.2f%%", change24h)
    }
}

struct Position: Identifiable {
    let id: String
    let symbol: String
    let quantity: Double
    let entryPrice: Double
    let currentPrice: Double
    let side: TradeSide
    
    var pnl: Double {
        switch side {
        case .long:
            return (currentPrice - entryPrice) * quantity
        case .short:
            return (entryPrice - currentPrice) * quantity
        }
    }
    
    var pnlPercent: Double {
        (pnl / (entryPrice * quantity)) * 100
    }
    
    var pnlColor: Color {
        pnl >= 0 ? .green : .red
    }
}

struct VoiceCommand: Identifiable {
    let id = UUID()
    let text: String
    let timestamp: Date
}

enum TradeSide {
    case long, short
}

enum SystemStatus {
    case ready, running, paused, stopped
    
    var color: Color {
        switch self {
        case .ready: return .gray
        case .running: return .green
        case .paused: return .orange
        case .stopped: return .red
        }
    }
    
    var icon: String {
        switch self {
        case .ready: return "circle"
        case .running: return "play.circle.fill"
        case .paused: return "pause.circle.fill"
        case .stopped: return "stop.circle.fill"
        }
    }
}