import SwiftUI
import Speech
import AVFoundation

struct AIChatView: View {
    @Environment(\.dismiss) var dismiss
    @StateObject private var chatManager = AIChatManager()
    @State private var messageText = ""
    @State private var isListening = false
    @FocusState private var isTextFieldFocused: Bool
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Chat Messages
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 16) {
                            ForEach(chatManager.messages) { message in
                                MessageBubble(message: message)
                                    .id(message.id)
                            }
                        }
                        .padding()
                    }
                    .onChange(of: chatManager.messages.count) { _ in
                        withAnimation {
                            proxy.scrollTo(chatManager.messages.last?.id, anchor: .bottom)
                        }
                    }
                }
                
                Divider()
                
                // Suggested Actions
                if !chatManager.suggestedActions.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(chatManager.suggestedActions, id: \.self) { action in
                                Button(action: {
                                    messageText = action
                                    sendMessage()
                                }) {
                                    Text(action)
                                        .font(.caption)
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 6)
                                        .background(Color.blue.opacity(0.2))
                                        .foregroundColor(.blue)
                                        .cornerRadius(12)
                                }
                            }
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 8)
                    }
                    
                    Divider()
                }
                
                // Input Area
                HStack(spacing: 12) {
                    // Voice Button
                    Button(action: toggleVoiceInput) {
                        Image(systemName: isListening ? "mic.fill" : "mic")
                            .foregroundColor(isListening ? .red : .blue)
                            .font(.title3)
                            .frame(width: 36, height: 36)
                            .background(isListening ? Color.red.opacity(0.2) : Color.blue.opacity(0.2))
                            .clipShape(Circle())
                            .overlay(
                                Circle()
                                    .stroke(isListening ? Color.red : Color.clear, lineWidth: 2)
                                    .scaleEffect(isListening ? 1.2 : 1)
                                    .opacity(isListening ? 0.5 : 1)
                                    .animation(
                                        isListening ? .easeInOut(duration: 1).repeatForever(autoreverses: true) : .default,
                                        value: isListening
                                    )
                            )
                    }
                    
                    // Text Field
                    TextField("Ask about your portfolio...", text: $messageText)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .focused($isTextFieldFocused)
                        .onSubmit {
                            sendMessage()
                        }
                    
                    // Send Button
                    Button(action: sendMessage) {
                        Image(systemName: "arrow.up.circle.fill")
                            .foregroundColor(messageText.isEmpty ? .gray : .blue)
                            .font(.title2)
                    }
                    .disabled(messageText.isEmpty)
                }
                .padding()
            }
            .navigationTitle("AI Assistant")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .onAppear {
                chatManager.addWelcomeMessage()
            }
        }
    }
    
    func sendMessage() {
        guard !messageText.isEmpty else { return }
        chatManager.sendMessage(messageText)
        messageText = ""
        isTextFieldFocused = false
    }
    
    func toggleVoiceInput() {
        if isListening {
            chatManager.stopListening()
            isListening = false
        } else {
            chatManager.startListening { text in
                messageText = text
            }
            isListening = true
        }
    }
}

// MARK: - Message Bubble
struct MessageBubble: View {
    let message: ChatMessage
    
    var body: some View {
        HStack {
            if message.isUser {
                Spacer()
            }
            
            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 4) {
                if let title = message.title {
                    Text(title)
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(message.isUser ? .white : .primary)
                }
                
                Text(message.text)
                    .padding(12)
                    .background(
                        message.isUser
                            ? Color.blue
                            : Color(.systemGray5)
                    )
                    .foregroundColor(message.isUser ? .white : .primary)
                    .cornerRadius(16)
                
                if let actions = message.actions, !actions.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(actions, id: \.title) { action in
                            Button(action: action.action) {
                                HStack {
                                    Image(systemName: action.icon)
                                    Text(action.title)
                                    Spacer()
                                }
                                .padding(8)
                                .background(Color.blue.opacity(0.1))
                                .foregroundColor(.blue)
                                .cornerRadius(8)
                            }
                        }
                    }
                    .padding(.top, 4)
                }
                
                Text(message.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: 280, alignment: message.isUser ? .trailing : .leading)
            
            if !message.isUser {
                Spacer()
            }
        }
    }
}

// MARK: - AI Chat Manager
class AIChatManager: NSObject, ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var suggestedActions = [
        "What's my current P&L?",
        "Show me BTC price",
        "Buy 0.1 BTC",
        "What's the market sentiment?",
        "Analyze my portfolio"
    ]
    
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    
    func addWelcomeMessage() {
        let welcomeMessage = ChatMessage(
            text: "Hello! I'm your AI trading assistant. I can help you with:\n\n• Portfolio analysis\n• Market insights\n• Trade execution\n• Risk management\n\nWhat would you like to know?",
            isUser: false,
            title: "AI Assistant"
        )
        messages.append(welcomeMessage)
    }
    
    func sendMessage(_ text: String) {
        // Add user message
        let userMessage = ChatMessage(text: text, isUser: true)
        messages.append(userMessage)
        
        // Simulate AI response
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) { [weak self] in
            let response = self?.generateAIResponse(for: text) ?? "I'm processing your request..."
            let aiMessage = ChatMessage(
                text: response.text,
                isUser: false,
                title: "AI Assistant",
                actions: response.actions
            )
            self?.messages.append(aiMessage)
        }
    }
    
    func generateAIResponse(for query: String) -> (text: String, actions: [ChatAction]?) {
        let lowercased = query.lowercased()
        
        if lowercased.contains("p&l") || lowercased.contains("profit") {
            return (
                text: "Your current portfolio shows:\n\n📈 Total P&L: +$25,430.50 (+25.43%)\n📊 Today's P&L: +$2,340.50 (+1.90%)\n💰 Win Rate: 68.5%\n\nYou're outperforming the market by 12.3%!",
                actions: [
                    ChatAction(title: "View Details", icon: "chart.line.uptrend.xyaxis") {
                        // Navigate to portfolio
                    },
                    ChatAction(title: "Export Report", icon: "square.and.arrow.up") {
                        // Export report
                    }
                ]
            )
        } else if lowercased.contains("btc") || lowercased.contains("bitcoin") {
            return (
                text: "Bitcoin (BTC-USD) is currently trading at:\n\n💰 Price: $110,768.89\n📉 24h Change: -1.95%\n📊 Volume: $28.5B\n\nTechnical indicators suggest a neutral stance with RSI at 52.",
                actions: [
                    ChatAction(title: "Buy BTC", icon: "plus.circle") {
                        // Open buy dialog
                    },
                    ChatAction(title: "Set Alert", icon: "bell") {
                        // Set price alert
                    }
                ]
            )
        } else if lowercased.contains("buy") {
            return (
                text: "I'll help you place a buy order. Please specify:\n\n• Symbol (e.g., BTC-USD)\n• Quantity\n• Order type (Market/Limit)\n\nOr use the quick action below:",
                actions: [
                    ChatAction(title: "Open Order Form", icon: "plus.square") {
                        // Open trading sheet
                    }
                ]
            )
        } else if lowercased.contains("sentiment") || lowercased.contains("market") {
            return (
                text: "Market Sentiment Analysis:\n\n🟢 Overall: Slightly Bullish\n😨 Fear & Greed: 52 (Neutral)\n📰 News Sentiment: Positive\n🐦 Social Media: Mixed\n\nTop trending: Bitcoin ETF approval speculation",
                actions: nil
            )
        } else if lowercased.contains("analyze") || lowercased.contains("portfolio") {
            return (
                text: "Portfolio Analysis:\n\n✅ Strengths:\n• Good diversification\n• Low correlation assets\n• Positive momentum\n\n⚠️ Considerations:\n• High crypto exposure (80%)\n• Consider taking profits on SOL\n• Add stop-losses to protect gains\n\n💡 Recommendation: Consider rebalancing to 60% crypto, 40% stablecoins",
                actions: [
                    ChatAction(title: "Rebalance Now", icon: "arrow.triangle.2.circlepath") {
                        // Rebalance portfolio
                    }
                ]
            )
        } else if lowercased.contains("help") {
            return (
                text: "I can help you with:\n\n📊 check portfolio\n📈 show [symbol] price\n💰 buy/sell [amount] [symbol]\n🎯 set alert for [symbol]\n📉 analyze performance\n🛡 check risk metrics\n📰 market news\n🤖 trading signals",
                actions: nil
            )
        } else {
            return (
                text: "I understand you're asking about \"\(query)\". Let me help you with that. Could you provide more details about what you'd like to know?",
                actions: nil
            )
        }
    }
    
    func startListening(completion: @escaping (String) -> Void) {
        // Request speech recognition permission
        SFSpeechRecognizer.requestAuthorization { authStatus in
            guard authStatus == .authorized else { return }
            
            DispatchQueue.main.async {
                self.startRecording(completion: completion)
            }
        }
    }
    
    func stopListening() {
        audioEngine.stop()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
    }
    
    private func startRecording(completion: @escaping (String) -> Void) {
        // Setup audio session
        let audioSession = AVAudioSession.sharedInstance()
        try? audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
        try? audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        
        let inputNode = audioEngine.inputNode
        guard let recognitionRequest = recognitionRequest else { return }
        
        recognitionRequest.shouldReportPartialResults = true
        
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { result, error in
            if let result = result {
                completion(result.bestTranscription.formattedString)
            }
            
            if error != nil || result?.isFinal == true {
                self.stopListening()
            }
        }
        
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }
        
        audioEngine.prepare()
        try? audioEngine.start()
    }
}

// MARK: - Chat Models
struct ChatMessage: Identifiable {
    let id = UUID()
    let text: String
    let isUser: Bool
    let timestamp = Date()
    var title: String? = nil
    var actions: [ChatAction]? = nil
}

struct ChatAction {
    let title: String
    let icon: String
    let action: () -> Void
}

// MARK: - Preview
struct AIChatView_Previews: PreviewProvider {
    static var previews: some View {
        AIChatView()
            .preferredColorScheme(.dark)
    }
}