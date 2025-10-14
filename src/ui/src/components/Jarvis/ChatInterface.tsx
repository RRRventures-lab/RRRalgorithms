import React, { useState, useRef, useEffect } from 'react';
import { jarvisService } from '../../services/jarvisService';

interface Message {
  id: string;
  type: 'user' | 'jarvis';
  content: string;
  timestamp: Date;
  isTyping?: boolean;
}

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isConnected, setIsConnected] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Add welcome message
    const welcomeMessage: Message = {
      id: 'welcome',
      type: 'jarvis',
      content: "Hello! I'm Jarvis, your AI trading assistant. I can help you with market analysis, portfolio management, trade execution, and more. What would you like to know?",
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
  }, []);

  const sendMessage = async () => {
    if (!inputValue.trim() || isTyping) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);
    setIsConnected(true);

    try {
      // Add typing indicator
      const typingMessage: Message = {
        id: `typing-${Date.now()}`,
        type: 'jarvis',
        content: '',
        timestamp: new Date(),
        isTyping: true
      };
      setMessages(prev => [...prev, typingMessage]);

      // Process with Jarvis
      const response = await jarvisService.processTextCommand(userMessage.content);
      
      // Remove typing indicator and add response
      setMessages(prev => {
        const withoutTyping = prev.filter(msg => !msg.isTyping);
        const jarvisMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'jarvis',
          content: response,
          timestamp: new Date()
        };
        return [...withoutTyping, jarvisMessage];
      });
    } catch (error) {
      console.error('Error processing message:', error);
      setIsConnected(false);
      
      // Remove typing indicator and add error message
      setMessages(prev => {
        const withoutTyping = prev.filter(msg => !msg.isTyping);
        const errorMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'jarvis',
          content: "I'm having trouble processing that request right now. Please check your connection and try again.",
          timestamp: new Date()
        };
        return [...withoutTyping, errorMessage];
      });
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    jarvisService.clearHistory();
    
    // Add welcome message back
    const welcomeMessage: Message = {
      id: 'welcome',
      type: 'jarvis',
      content: "Hello! I'm Jarvis, your AI trading assistant. I can help you with market analysis, portfolio management, trade execution, and more. What would you like to know?",
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
  };

  const handleQuickCommand = async (command: string) => {
    setInputValue(command);
    // Small delay to ensure input is set
    setTimeout(() => {
      sendMessage();
    }, 100);
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getMessageColor = (type: 'user' | 'jarvis') => {
    return type === 'user' 
      ? 'bg-bloomberg-blue text-white' 
      : 'bg-terminal-border text-terminal-text';
  };

  const getMessageAlignment = (type: 'user' | 'jarvis') => {
    return type === 'user' ? 'justify-end' : 'justify-start';
  };

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="terminal-panel-header flex justify-between items-center p-1 border-b border-terminal-border">
        <span className="text-terminal-green text-terminal-sm font-bold">JARVIS CHAT</span>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-bloomberg-green' : 'bg-bloomberg-red'}`}></div>
          <span className="text-terminal-xs">
            {isConnected ? 'ONLINE' : 'OFFLINE'}
          </span>
          <button
            onClick={clearChat}
            className="bloomberg-button text-terminal-xs px-2 py-1"
          >
            CLEAR
          </button>
        </div>
      </div>
      
      <div className="terminal-content flex-1 flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto space-y-2 p-2 terminal-scrollbar">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${getMessageAlignment(message.type)}`}
            >
              <div
                className={`max-w-xs p-2 rounded ${
                  message.isTyping 
                    ? 'bg-terminal-border text-terminal-accent' 
                    : getMessageColor(message.type)
                }`}
              >
                {message.isTyping ? (
                  <div className="flex items-center space-x-1">
                    <div className="flex space-x-1">
                      <div className="w-1 h-1 bg-terminal-accent rounded-full animate-bounce"></div>
                      <div className="w-1 h-1 bg-terminal-accent rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-1 h-1 bg-terminal-accent rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                    <span className="text-terminal-xs ml-2">Jarvis is thinking...</span>
                  </div>
                ) : (
                  <>
                    <div className="text-terminal-xs">
                      {message.content}
                    </div>
                    <div className="text-terminal-xs opacity-70 mt-1">
                      {formatTime(message.timestamp)}
                    </div>
                  </>
                )}
              </div>
            </div>
          ))}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Commands */}
        <div className="border-t border-terminal-border p-2">
          <div className="text-terminal-accent text-terminal-xs font-bold mb-2">
            QUICK COMMANDS:
          </div>
          <div className="grid grid-cols-2 gap-1 mb-2">
            <button
              onClick={() => handleQuickCommand('What is my portfolio status?')}
              disabled={isTyping}
              className="bloomberg-button text-terminal-xs py-1 disabled:opacity-50"
            >
              Portfolio Status
            </button>
            <button
              onClick={() => handleQuickCommand('Show me Bitcoin price')}
              disabled={isTyping}
              className="bloomberg-button text-terminal-xs py-1 disabled:opacity-50"
            >
              Bitcoin Price
            </button>
            <button
              onClick={() => handleQuickCommand('What is the market doing today?')}
              disabled={isTyping}
              className="bloomberg-button text-terminal-xs py-1 disabled:opacity-50"
            >
              Market Analysis
            </button>
            <button
              onClick={() => handleQuickCommand('Show me recent trades')}
              disabled={isTyping}
              className="bloomberg-button text-terminal-xs py-1 disabled:opacity-50"
            >
              Recent Trades
            </button>
          </div>
        </div>

        {/* Input */}
        <div className="border-t border-terminal-border p-2">
          <div className="flex space-x-2">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask Jarvis anything..."
              className="flex-1 bg-terminal-bg border border-terminal-border text-terminal-text p-2 text-terminal-sm focus:outline-none focus:border-terminal-accent"
              disabled={isTyping}
            />
            <button
              onClick={sendMessage}
              disabled={!inputValue.trim() || isTyping}
              className="bloomberg-button px-3 py-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              SEND
            </button>
          </div>
        </div>

        {/* Connection Status */}
        {!isConnected && (
          <div className="bg-bloomberg-red bg-opacity-20 border-t border-bloomberg-red p-2">
            <div className="text-bloomberg-red text-terminal-xs text-center">
              Connection lost. Jarvis may not respond properly.
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatInterface;
