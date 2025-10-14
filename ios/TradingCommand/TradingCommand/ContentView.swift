import SwiftUI

struct ContentView: View {
    @EnvironmentObject var marketDataManager: MarketDataManager
    @EnvironmentObject var portfolioManager: PortfolioManager
    @EnvironmentObject var tradingManager: TradingManager
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            DashboardView()
                .tabItem {
                    Label("Dashboard", systemImage: "chart.line.uptrend.xyaxis")
                }
                .tag(0)
            
            TradingView()
                .tabItem {
                    Label("Trade", systemImage: "arrow.left.arrow.right")
                }
                .tag(1)
            
            BacktestingView()
                .tabItem {
                    Label("Backtest", systemImage: "clock.arrow.circlepath")
                }
                .tag(2)
            
            PortfolioView()
                .tabItem {
                    Label("Portfolio", systemImage: "briefcase.fill")
                }
                .tag(3)
            
            ControlsView()
                .tabItem {
                    Label("Controls", systemImage: "slider.horizontal.3")
                }
                .tag(4)
        }
        .accentColor(.green)
    }
}

// MARK: - Controls View
struct ControlsView: View {
    @EnvironmentObject var tradingManager: TradingManager
    @EnvironmentObject var voiceCommandManager: VoiceCommandManager
    @State private var showEmergencyConfirmation = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // System Status Card
                    StatusCard()
                    
                    // Trading Controls
                    TradingControlsCard()
                    
                    // Risk Management
                    RiskManagementCard()
                    
                    // Voice Commands
                    VoiceCommandCard()
                    
                    // Emergency Stop
                    EmergencyStopCard(showConfirmation: $showEmergencyConfirmation)
                }
                .padding()
            }
            .navigationTitle("Controls")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}

// MARK: - Status Card
struct StatusCard: View {
    @EnvironmentObject var tradingManager: TradingManager
    @EnvironmentObject var marketDataManager: MarketDataManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: tradingManager.systemStatus.icon)
                    .foregroundColor(tradingManager.systemStatus.color)
                    .font(.title2)
                
                Text("System Status")
                    .font(.headline)
                
                Spacer()
                
                Text(String(describing: tradingManager.systemStatus).uppercased())
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(tradingManager.systemStatus.color)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(tradingManager.systemStatus.color.opacity(0.2))
                    .cornerRadius(4)
            }
            
            Divider()
            
            HStack {
                StatusIndicator(
                    title: "WebSocket",
                    isConnected: marketDataManager.isConnected,
                    icon: "wifi"
                )
                
                Spacer()
                
                StatusIndicator(
                    title: "Trading",
                    isConnected: tradingManager.isTradingEnabled,
                    icon: "chart.line.uptrend.xyaxis"
                )
                
                Spacer()
                
                StatusIndicator(
                    title: "API",
                    isConnected: true,
                    icon: "server.rack"
                )
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct StatusIndicator: View {
    let title: String
    let isConnected: Bool
    let icon: String
    
    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: icon)
                .foregroundColor(isConnected ? .green : .red)
                .font(.title2)
            
            Text(title)
                .font(.caption2)
                .foregroundColor(.secondary)
            
            Circle()
                .fill(isConnected ? Color.green : Color.red)
                .frame(width: 8, height: 8)
                .overlay(
                    Circle()
                        .stroke(Color.white.opacity(0.8), lineWidth: 1)
                        .scaleEffect(isConnected ? 2 : 1)
                        .opacity(isConnected ? 0 : 1)
                        .animation(
                            isConnected ? .easeInOut(duration: 1).repeatForever(autoreverses: false) : .default,
                            value: isConnected
                        )
                )
        }
    }
}

// MARK: - Trading Controls Card
struct TradingControlsCard: View {
    @EnvironmentObject var tradingManager: TradingManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Trading Controls")
                .font(.headline)
            
            HStack(spacing: 12) {
                ControlButton(
                    title: "Start",
                    icon: "play.fill",
                    color: .green,
                    action: {
                        tradingManager.systemStatus = .running
                        tradingManager.isTradingEnabled = true
                    }
                )
                
                ControlButton(
                    title: "Pause",
                    icon: "pause.fill",
                    color: .orange,
                    action: {
                        tradingManager.systemStatus = .paused
                        tradingManager.isTradingEnabled = false
                    }
                )
                
                ControlButton(
                    title: "Stop",
                    icon: "stop.fill",
                    color: .red,
                    action: {
                        tradingManager.systemStatus = .stopped
                        tradingManager.isTradingEnabled = false
                    }
                )
            }
            
            Toggle("Auto Trading", isOn: $tradingManager.isTradingEnabled)
                .toggleStyle(SwitchToggleStyle(tint: .green))
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct ControlButton: View {
    let title: String
    let icon: String
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.title2)
                Text(title)
                    .font(.caption)
                    .fontWeight(.medium)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .foregroundColor(.white)
            .background(color)
            .cornerRadius(8)
        }
    }
}

// MARK: - Risk Management Card
struct RiskManagementCard: View {
    @EnvironmentObject var tradingManager: TradingManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Risk Management")
                .font(.headline)
            
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Risk Level")
                        .font(.subheadline)
                    Spacer()
                    Text("\(Int(tradingManager.riskLevel))%")
                        .font(.subheadline)
                        .fontWeight(.bold)
                        .foregroundColor(riskColor)
                }
                
                Slider(value: $tradingManager.riskLevel, in: 0...100, step: 5)
                    .accentColor(riskColor)
                
                HStack {
                    Text("Conservative")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("Moderate")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("Aggressive")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            
            Divider()
            
            HStack {
                Text("Max Position Size")
                    .font(.subheadline)
                Spacer()
                Text("$\(Int(tradingManager.maxPositionSize))")
                    .font(.subheadline)
                    .fontWeight(.bold)
            }
            
            Stepper("", value: $tradingManager.maxPositionSize, in: 100...10000, step: 100)
                .labelsHidden()
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    var riskColor: Color {
        if tradingManager.riskLevel <= 30 {
            return .green
        } else if tradingManager.riskLevel <= 60 {
            return .orange
        } else {
            return .red
        }
    }
}

// MARK: - Voice Command Card
struct VoiceCommandCard: View {
    @EnvironmentObject var voiceCommandManager: VoiceCommandManager
    @State private var isRecording = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Voice Commands")
                    .font(.headline)
                Spacer()
                Image(systemName: "mic.fill")
                    .foregroundColor(isRecording ? .red : .gray)
            }
            
            Text("Say \"Hey Trading\" to activate")
                .font(.caption)
                .foregroundColor(.secondary)
            
            HStack {
                Button(action: {
                    isRecording.toggle()
                    if isRecording {
                        voiceCommandManager.startListening()
                    } else {
                        voiceCommandManager.stopListening()
                    }
                }) {
                    HStack {
                        Image(systemName: isRecording ? "mic.slash.fill" : "mic.fill")
                        Text(isRecording ? "Stop Listening" : "Start Listening")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .foregroundColor(.white)
                    .background(isRecording ? Color.red : Color.blue)
                    .cornerRadius(8)
                }
            }
            
            if !voiceCommandManager.lastCommand.isEmpty {
                Text("Last command: \"\(voiceCommandManager.lastCommand)\"")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .italic()
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Emergency Stop Card
struct EmergencyStopCard: View {
    @EnvironmentObject var tradingManager: TradingManager
    @Binding var showConfirmation: Bool
    
    var body: some View {
        VStack(spacing: 12) {
            Button(action: {
                showConfirmation = true
            }) {
                HStack {
                    Image(systemName: "exclamationmark.octagon.fill")
                        .font(.title2)
                    Text("EMERGENCY STOP")
                        .font(.headline)
                        .fontWeight(.bold)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .foregroundColor(.white)
                .background(Color.red)
                .cornerRadius(12)
            }
            .alert("Emergency Stop", isPresented: $showConfirmation) {
                Button("Cancel", role: .cancel) { }
                Button("STOP ALL TRADING", role: .destructive) {
                    tradingManager.emergencyStop()
                }
            } message: {
                Text("This will immediately close all positions and stop all trading activities. Are you sure?")
            }
            
            Text("Closes all positions immediately")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Alerts View
struct AlertsView: View {
    @State private var alerts: [TradingAlert] = TradingAlert.mockAlerts
    
    var body: some View {
        NavigationView {
            List {
                ForEach(alerts) { alert in
                    AlertRow(alert: alert)
                }
                .onDelete(perform: deleteAlerts)
            }
            .navigationTitle("Alerts")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Clear All") {
                        alerts.removeAll()
                    }
                }
            }
        }
    }
    
    func deleteAlerts(at offsets: IndexSet) {
        alerts.remove(atOffsets: offsets)
    }
}

struct AlertRow: View {
    let alert: TradingAlert
    
    var body: some View {
        HStack {
            Image(systemName: alert.icon)
                .foregroundColor(alert.color)
                .font(.title3)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(alert.title)
                    .font(.headline)
                Text(alert.message)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                Text(alert.timestamp, style: .relative)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
        }
        .padding(.vertical, 4)
    }
}

struct TradingAlert: Identifiable {
    let id = UUID()
    let title: String
    let message: String
    let severity: AlertSeverity
    let timestamp: Date
    
    var icon: String {
        switch severity {
        case .info: return "info.circle.fill"
        case .warning: return "exclamationmark.triangle.fill"
        case .error: return "xmark.octagon.fill"
        case .success: return "checkmark.circle.fill"
        }
    }
    
    var color: Color {
        switch severity {
        case .info: return .blue
        case .warning: return .orange
        case .error: return .red
        case .success: return .green
        }
    }
    
    static var mockAlerts: [TradingAlert] {
        [
            TradingAlert(title: "Trade Executed", message: "BTC-USD: Buy 0.1 @ $110,768.89", severity: .success, timestamp: Date().addingTimeInterval(-60)),
            TradingAlert(title: "Price Alert", message: "ETH-USD crossed above $3,750", severity: .info, timestamp: Date().addingTimeInterval(-300)),
            TradingAlert(title: "Risk Warning", message: "Portfolio drawdown approaching 5%", severity: .warning, timestamp: Date().addingTimeInterval(-600)),
            TradingAlert(title: "Connection Lost", message: "WebSocket disconnected, retrying...", severity: .error, timestamp: Date().addingTimeInterval(-900))
        ]
    }
}

enum AlertSeverity {
    case info, warning, error, success
}

// MARK: - Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .environmentObject(MarketDataManager())
            .environmentObject(PortfolioManager())
            .environmentObject(TradingManager())
            .environmentObject(VoiceCommandManager())
            .preferredColorScheme(.dark)
    }
}