import React, { useState, useRef, useEffect } from 'react';
import { jarvisService } from '../../services/jarvisService';

const VoiceInput: React.FC = () => {
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastResponse, setLastResponse] = useState('');
  const [transcription, setTranscription] = useState('');
  const [error, setError] = useState('');
  const recognitionRef = useRef<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    // Initialize speech recognition
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setTranscription(transcript);
        processVoiceCommand(transcript);
      };

      recognitionRef.current.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setError(`Speech recognition error: ${event.error}`);
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    } else {
      setError('Speech recognition not supported in this browser');
    }

    // Initialize audio context for TTS
    try {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    } catch (error) {
      console.warn('Audio context not available:', error);
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  const startListening = () => {
    if (!recognitionRef.current) {
      setError('Speech recognition not available');
      return;
    }

    setError('');
    setTranscription('');
    setIsListening(true);
    
    try {
      recognitionRef.current.start();
    } catch (error) {
      console.error('Error starting speech recognition:', error);
      setError('Failed to start speech recognition');
      setIsListening(false);
    }
  };

  const stopListening = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    setIsListening(false);
  };

  const processVoiceCommand = async (transcript: string) => {
    setIsProcessing(true);
    setError('');

    try {
      const response = await jarvisService.processTextCommand(transcript);
      setLastResponse(response);
      
      // Speak the response
      await speakText(response);
    } catch (error) {
      console.error('Error processing voice command:', error);
      setError('Failed to process voice command');
    } finally {
      setIsProcessing(false);
    }
  };

  const speakText = async (text: string): Promise<void> => {
    return new Promise((resolve, reject) => {
      if (!('speechSynthesis' in window)) {
        console.warn('Speech synthesis not supported');
        resolve();
        return;
      }

      // Stop any current speech
      speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 0.8;
      utterance.volume = 0.8;

      // Use a male voice
      const voices = speechSynthesis.getVoices();
      const maleVoice = voices.find(voice => 
        voice.name.includes('Daniel') || 
        voice.name.includes('Alex') ||
        voice.name.includes('Male') ||
        voice.name.includes('Google UK English Male') ||
        voice.name.includes('Microsoft David')
      );

      if (maleVoice) {
        utterance.voice = maleVoice;
      }

      utterance.onend = () => resolve();
      utterance.onerror = (event) => {
        console.error('TTS error:', event);
        reject(new Error(`TTS error: ${event.error}`));
      };

      speechSynthesis.speak(utterance);
    });
  };

  const handleQuickCommand = async (command: string) => {
    setIsProcessing(true);
    setError('');

    try {
      const response = await jarvisService.processTextCommand(command);
      setLastResponse(response);
      await speakText(response);
    } catch (error) {
      console.error('Error processing quick command:', error);
      setError('Failed to process command');
    } finally {
      setIsProcessing(false);
    }
  };

  const clearResponse = () => {
    setLastResponse('');
    setTranscription('');
    setError('');
  };

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="terminal-panel-header flex justify-between items-center p-1 border-b border-terminal-border">
        <span className="text-terminal-green text-terminal-sm font-bold">JARVIS VOICE ASSISTANT</span>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${
            isListening ? 'bg-bloomberg-red animate-pulse' : 
            isProcessing ? 'bg-bloomberg-amber animate-pulse' : 
            'bg-bloomberg-green'
          }`}></div>
          <span className="text-terminal-xs">
            {isListening ? 'LISTENING' : isProcessing ? 'PROCESSING' : 'READY'}
          </span>
        </div>
      </div>
      
      <div className="terminal-content flex-1 p-2 space-y-3">
        {/* Voice Input Controls */}
        <div className="flex items-center justify-center">
          <button
            onClick={isListening ? stopListening : startListening}
            disabled={isProcessing}
            className={`bloomberg-button flex items-center space-x-2 px-4 py-2 ${
              isListening ? 'bg-bloomberg-red hover:bg-red-600' : 
              'bg-bloomberg-green hover:bg-green-600'
            } ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <span className="text-2xl">
              {isListening ? '⏹️' : '🎤'}
            </span>
            <span className="text-terminal-sm font-bold">
              {isListening ? 'Stop Listening' : 'Start Voice Command'}
            </span>
          </button>
        </div>

        {/* Status Display */}
        <div className="text-center">
          <div className="text-terminal-xs text-terminal-accent">
            {isListening && "🎤 Listening... Speak your command"}
            {isProcessing && "🤖 Processing your request..."}
            {!isListening && !isProcessing && "Press the microphone to talk to Jarvis"}
          </div>
        </div>

        {/* Transcription */}
        {transcription && (
          <div className="bg-terminal-border p-2 rounded">
            <div className="text-terminal-accent text-terminal-xs font-bold mb-1">
              YOU SAID:
            </div>
            <div className="text-terminal-text text-terminal-sm">
              "{transcription}"
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="bg-bloomberg-red bg-opacity-20 border border-bloomberg-red p-2 rounded">
            <div className="text-bloomberg-red text-terminal-xs font-bold mb-1">
              ERROR:
            </div>
            <div className="text-terminal-text text-terminal-sm">
              {error}
            </div>
          </div>
        )}

        {/* Last Response */}
        {lastResponse && (
          <div className="bg-terminal-border p-3 rounded flex-1">
            <div className="flex justify-between items-center mb-2">
              <div className="text-terminal-accent text-terminal-xs font-bold">
                JARVIS RESPONSE:
              </div>
              <button
                onClick={clearResponse}
                className="bloomberg-button text-terminal-xs px-2 py-1"
              >
                CLEAR
              </button>
            </div>
            <div className="text-terminal-text text-terminal-sm max-h-32 overflow-y-auto terminal-scrollbar">
              {lastResponse}
            </div>
          </div>
        )}

        {/* Quick Commands */}
        <div className="space-y-2">
          <div className="text-terminal-accent text-terminal-xs font-bold">
            QUICK COMMANDS:
          </div>
          <div className="grid grid-cols-2 gap-2">
            <button 
              onClick={() => handleQuickCommand('What is my portfolio status?')}
              disabled={isProcessing}
              className="bloomberg-button text-terminal-xs py-1 disabled:opacity-50"
            >
              Portfolio Status
            </button>
            <button 
              onClick={() => handleQuickCommand('Show me Bitcoin price')}
              disabled={isProcessing}
              className="bloomberg-button text-terminal-xs py-1 disabled:opacity-50"
            >
              Bitcoin Price
            </button>
            <button 
              onClick={() => handleQuickCommand('What is the market doing today?')}
              disabled={isProcessing}
              className="bloomberg-button text-terminal-xs py-1 disabled:opacity-50"
            >
              Market Analysis
            </button>
            <button 
              onClick={() => handleQuickCommand('Show me recent trades')}
              disabled={isProcessing}
              className="bloomberg-button text-terminal-xs py-1 disabled:opacity-50"
            >
              Recent Trades
            </button>
          </div>
        </div>

        {/* Voice Commands Help */}
        <div className="space-y-1">
          <div className="text-terminal-accent text-terminal-xs font-bold">
            VOICE COMMANDS:
          </div>
          <div className="text-terminal-xs text-terminal-text space-y-1">
            <div>• "Buy 0.1 Bitcoin"</div>
            <div>• "Sell 2 Ethereum"</div>
            <div>• "What's my portfolio worth?"</div>
            <div>• "Show me Solana price"</div>
            <div>• "How are my trades doing?"</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VoiceInput;
