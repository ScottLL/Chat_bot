import React, { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

interface ApiResponse {
  type: 'status' | 'metadata' | 'word' | 'complete' | 'error' | 'heartbeat';
  // Status message
  status?: string;
  details?: {
    session_id?: string;
    capabilities?: string[];
    protocol_version?: string;
  };
  // Metadata message
  confidence?: number;
  total_words?: number;
  session_id?: string;
  // Word message
  content?: string;
  index?: number;
  // Complete message
  processing_stage?: string;
  final_confidence?: number;
  // Error message
  error?: string;
  // Heartbeat
  timestamp?: number;
}

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<string>('disconnected');
  const [sessionId, setSessionId] = useState<string | null>(null);
  
  const answerRef = useRef<HTMLDivElement>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  // Auto-scroll to bottom of answer area
  useEffect(() => {
    if (answerRef.current) {
      answerRef.current.scrollTop = answerRef.current.scrollHeight;
    }
  }, [answer]);

  // Test API connection on component mount
  useEffect(() => {
    testApiConnection();
  }, []);

  const testApiConnection = useCallback(async () => {
    try {
      setConnectionStatus('connecting');
      // Use the backend URL - prioritize environment variable, fallback to deployed backend
      const apiUrl = process.env.REACT_APP_API_URL || 'https://defi-qa-backend.fly.dev';
      console.log('🔗 Testing API connection to:', apiUrl);
      const response = await fetch(`${apiUrl}/health`);
      
      if (response.ok) {
        const data = await response.json();
        console.log('✅ API Health Check:', data);
        setConnectionStatus('connected');
        setError(null);
        reconnectAttempts.current = 0;
        
        // Generate a client-side session ID for tracking
        setSessionId(`client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (connectionError) {
      console.error('❌ API Connection failed:', connectionError);
      setConnectionStatus('error');
      setError(connectionError instanceof Error ? connectionError.message : 'Unknown connection error');
    }
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }

    if (connectionStatus !== 'connected') {
      setError('Not connected to server. Please wait for connection...');
      testApiConnection();
      return;
    }

    // Reset state
    setAnswer('');
    setError(null);
    setConfidence(null);
    setIsLoading(true);

    try {
      // Use HTTP with Server-Sent Events instead of WebSocket
      const apiUrl = process.env.REACT_APP_API_URL || 'https://defi-qa-backend.fly.dev';
      console.log('🔗 Sending question to:', `${apiUrl}/v2/ask-stream`);
      
      const response = await fetch(`${apiUrl}/v2/ask-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({ question: question.trim() }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      if (!response.body) {
        throw new Error('No response body received');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let currentAnswer = '';

      try {
        while (true) {
          const { done, value } = await reader.read();
          
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ') && line.length > 6) {
              try {
                const data = JSON.parse(line.slice(6));
                
                if (data.type === 'word' && data.content) {
                  currentAnswer += (currentAnswer ? ' ' : '') + data.content;
                  setAnswer(currentAnswer);
                  if (data.confidence !== undefined) {
                    setConfidence(data.confidence);
                  }
                } else if (data.type === 'complete') {
                  console.log('✅ Streaming complete:', data.processing_stage);
                  if (data.final_confidence !== undefined) {
                    setConfidence(data.final_confidence);
                  }
                  setIsLoading(false);
                  break;
                } else if (data.type === 'error') {
                  throw new Error(data.error || 'Stream processing error');
                }
              } catch (parseError) {
                console.warn('Failed to parse SSE data:', line, parseError);
              }
            }
          }
        }
      } finally {
        reader.releaseLock();
      }

    } catch (sendError) {
      console.error('Failed to send question:', sendError);
      setError(`Failed to process question: ${sendError instanceof Error ? sendError.message : 'Unknown error'}`);
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  const clearAll = () => {
    setQuestion('');
    setAnswer('');
    setError(null);
    setConfidence(null);
    setIsLoading(false);
  };

  const manualReconnect = () => {
    console.log('🔄 Manual reconnection requested');
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    reconnectAttempts.current = 0;
    setTimeout(() => {
      testApiConnection();
    }, 1000);
  };

  const testApiEndpoint = async () => {
    console.log('🧪 Testing API endpoint...');
    try {
      const apiUrl = process.env.REACT_APP_API_URL || 'https://defi-qa-backend.fly.dev';
      console.log('🔗 Testing endpoint:', `${apiUrl}/ask`);
      const response = await fetch(`${apiUrl}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: 'What is DeFi?' }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ API Test successful:', data);
        alert('API test successful! Check console for details.');
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (testError) {
      console.error('❌ API Test failed:', testError);
      alert(`API test failed: ${testError}`);
    }
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return '#10b981';
      case 'connecting': return '#f59e0b';
      case 'error': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Connected';
      case 'connecting': return 'Connecting...';
      case 'error': return 'Connection Error';
      default: return 'Disconnected';
    }
  };

  return (
    <div className="app">
      <div className="container">
        {/* Header */}
        <header className="header">
          <h1 className="title">
            <span className="title-main">Chaos Labs DeFi Q&A</span>
            <span className="title-subtitle">Instant answers to your DeFi questions</span>
          </h1>

          {/* Connection Status */}
          <div className="connection-status">
            <div 
              className="status-indicator"
              style={{ backgroundColor: getConnectionStatusColor() }}
            ></div>
            <span className="status-text">{getConnectionStatusText()}</span>
            {sessionId && (
              <span className="session-id">Session: {sessionId.slice(0, 8)}...</span>
            )}
            {(connectionStatus === 'error' || connectionStatus === 'disconnected') && (
              <>
                <button 
                  onClick={manualReconnect}
                  className="btn btn-secondary"
                  style={{ marginLeft: '10px', padding: '4px 8px', fontSize: '12px' }}
                >
                  Reconnect
                </button>
                <button 
                  onClick={testApiEndpoint}
                  className="btn btn-secondary"
                  style={{ marginLeft: '5px', padding: '4px 8px', fontSize: '12px' }}
                >
                  Test API
                </button>
              </>
            )}
          </div>
        </header>

        {/* Main Content */}
        <main className="main">
          {/* Question Input */}
          <form onSubmit={handleSubmit} className="question-form">
            <div className="input-container">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask me anything about DeFi... (e.g., 'What is yield farming?')"
                className="question-input"
                disabled={isLoading || connectionStatus !== 'connected'}
                rows={2}
                maxLength={500}
              />
              <div className="input-actions">
                <span className="char-count">
                  {question.length}/500
                </span>
                <div className="button-group">
                  {(answer || question) && (
                    <button 
                      type="button"
                      onClick={clearAll}
                      className="btn btn-secondary"
                      disabled={isLoading}
                    >
                      Clear
                    </button>
                  )}
                  <button 
                    type="submit"
                    disabled={isLoading || !question.trim() || connectionStatus !== 'connected'}
                    className="btn btn-primary"
                  >
                    {isLoading ? (
                      <>
                        <span className="loading-spinner"></span>
                        Searching...
                      </>
                    ) : connectionStatus !== 'connected' ? (
                      'Connecting...'
                    ) : (
                      'Ask Question'
                    )}
                  </button>
                </div>
              </div>
            </div>
          </form>

          {/* Answer Area */}
          {(answer || isLoading || error) && (
            <div className="answer-section">
              <div className="answer-header">
                <h2>Answer</h2>
                {confidence !== null && !error && (
                  <div className="confidence-badge">
                    <span className="confidence-label">Confidence:</span>
                    <span className="confidence-value">
                      {Math.round(confidence * 100)}%
                    </span>
                    <div 
                      className="confidence-bar"
                      style={{
                        background: `linear-gradient(90deg, 
                          ${confidence > 0.8 ? '#10b981' : 
                            confidence > 0.6 ? '#f59e0b' : '#ef4444'} 
                          ${confidence * 100}%, 
                          #e5e7eb ${confidence * 100}%)`
                      }}
                    ></div>
                  </div>
                )}
              </div>

              <div 
                ref={answerRef}
                className={`answer-content ${isLoading ? 'loading' : ''} ${error ? 'error' : ''}`}
              >
                {error ? (
                  <div className="error-message">
                    <span className="error-icon">⚠️</span>
                    <div>
                      <p><strong>Something went wrong:</strong></p>
                      <p>{error}</p>
                      <div className="error-suggestions">
                        {error.includes('Connection') || error.includes('connect') ? (
                          <>
                            <p className="error-suggestion"><strong>Connection Issue:</strong></p>
                            <ul>
                              <li>Check that the API server is running at <code>defi-qa-backend.fly.dev</code></li>
                              <li>Verify your internet connection</li>
                              <li>Try refreshing the page</li>
                              <li>API connection may need time to establish</li>
                            </ul>
                          </>
                        ) : error.includes('No match') || error.includes('no results') ? (
                          <>
                            <p className="error-suggestion"><strong>No Match Found:</strong></p>
                            <ul>
                              <li>Try rephrasing your question</li>
                              <li>Ask about DeFi topics like lending, staking, or yield farming</li>
                              <li>Use more specific terms related to DeFi protocols</li>
                            </ul>
                          </>
                        ) : error.includes("too short") || error.includes("Question is") ? (
                          <>
                            <p className="error-suggestion"><strong>Question Issue:</strong></p>
                            <ul>
                              <li>Please provide a more detailed question</li>
                              <li>Include specific DeFi terms or protocol names</li>
                              <li>Ask about concepts like APY, liquidity pools, or smart contracts</li>
                            </ul>
                          </>
                        ) : (
                          <>
                            <p className="error-suggestion"><strong>Try These Steps:</strong></p>
                            <ul>
                              <li>Check your connection and try again</li>
                              <li>Refresh the page and retry</li>
                              <li>Contact support if the issue persists</li>
                            </ul>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <>
                    {answer && (
                      <div className="answer-display">
                        {(() => {
                          // Parse answer to separate main content from references
                          // Handle actual newlines from streaming response
                          const referenceStartIndex = answer.search(/\n\n📚 \*\*Reference\*\*/);
                          
                          if (referenceStartIndex !== -1) {
                            // Split main answer from references
                            const mainAnswer = answer.substring(0, referenceStartIndex).trim();
                            const referencesSection = answer.substring(referenceStartIndex + 2).trim();
                            
                            // Extract all reference lines
                            const referenceLines = referencesSection
                              .split('\n')
                              .filter(line => line.trim().startsWith('📚 **Reference**:'))
                              .map(line => {
                                const match = line.match(/📚 \*\*Reference\*\*: (.+)$/);
                                return match ? match[1].trim() : line.trim();
                              });
                            
                            return (
                              <>
                                <div className="answer-text">
                                  {mainAnswer}
                                </div>
                                
                                {referenceLines.length > 0 && (
                                  <div className="answer-reference">
                                    {referenceLines.length === 1 ? (
                                      <span>
                                        <span className="reference-icon">📚</span>
                                        {referenceLines[0]}
                                      </span>
                                    ) : (
                                      <>
                                        <div style={{ marginBottom: '0.25rem', fontWeight: 'bold' }}>
                                          <span className="reference-icon">📚</span>
                                          References:
                                        </div>
                                        {referenceLines.map((ref, index) => (
                                          <div key={index} style={{ marginLeft: '1rem' }}>
                                            • {ref}
                                          </div>
                                        ))}
                                      </>
                                    )}
                                  </div>
                                )}
                              </>
                            );
                          }
                          
                          // Fallback: Check for references that might not have proper double newlines
                          else if (answer.includes('📚 **Reference**:')) {
                            const referenceRegex = /📚 \*\*Reference\*\*: (.+?)(?=📚 \*\*Reference\*\*:|$)/g;
                            const references = [];
                            let match;
                            
                            while ((match = referenceRegex.exec(answer)) !== null) {
                              references.push(match[1].trim());
                            }
                            
                            if (references.length > 0) {
                              // Remove all reference text from main answer
                              const mainAnswer = answer
                                .replace(/📚 \*\*Reference\*\*: .+/g, '')
                                .trim();
                              
                              return (
                                <>
                                  <div className="answer-text">
                                    {mainAnswer}
                                  </div>
                                  
                                  <div className="answer-reference">
                                    {references.length === 1 ? (
                                      <span>
                                        <span className="reference-icon">📚</span>
                                        {references[0]}
                                      </span>
                                    ) : (
                                      <>
                                        <div style={{ marginBottom: '0.25rem', fontWeight: 'bold' }}>
                                          <span className="reference-icon">📚</span>
                                          References:
                                        </div>
                                        {references.map((ref, index) => (
                                          <div key={index} style={{ marginLeft: '1rem' }}>
                                            • {ref}
                                          </div>
                                        ))}
                                      </>
                                    )}
                                  </div>
                                </>
                              );
                            }
                          }
                          
                          // No references found, display normal answer
                          return (
                            <div className="answer-text">
                              {answer}
                            </div>
                          );
                        })()}
                      </div>
                    )}
                    
                    {isLoading && !answer && (
                      <div className="loading-state">
                        <div className="loading-dots">
                          <span></span>
                          <span></span>
                          <span></span>
                        </div>
                        <p>Searching through DeFi knowledge base...</p>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          )}

          {/* Sample Questions */}
          {!answer && !isLoading && !error && connectionStatus === 'connected' && (
            <div className="sample-questions">
              <h3>Try asking about:</h3>
              <div className="question-tags">
                {[
                  "What is yield farming?",
                  "How does Aave lending work?", 
                  "What are impermanent losses?",
                  "Explain DeFi liquidity pools",
                  "What is the largest lending pool on Aave?"
                ].map((sampleQ, index) => (
                  <button
                    key={index}
                    className="question-tag"
                    onClick={() => setQuestion(sampleQ)}
                    disabled={isLoading || connectionStatus !== 'connected'}
                  >
                    {sampleQ}
                  </button>
                ))}
              </div>
            </div>
          )}
        </main>

        {/* Footer */}
        <footer className="footer">
          <p>
            Powered by AI • {" "}
            <a 
              href="https://openai.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="footer-link"
            >
              OpenAI Embeddings
            </a>
            {" "} • {" "}
            <a 
              href="https://python.langchain.com/docs/langgraph" 
              target="_blank" 
              rel="noopener noreferrer"
              className="footer-link"
            >
              LangGraph
            </a>
            {" "} • HTTP Streaming
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
