import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8002';

function App() {
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [user, setUser] = useState(JSON.parse(localStorage.getItem('user') || 'null'));
  const [isLogin, setIsLogin] = useState(true);

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');

  const [files, setFiles] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (token && user) {
      fetchFiles();
      fetchSessions();
    }
  }, [token, user]);

  useEffect(() => {
    if (selectedSession) {
      setMessages([]); // Clear immediately to prevent flash
      fetchMessages();
    } else {
      setMessages([]);
    }
  }, [selectedSession]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleAuth = async (e) => {
    e.preventDefault();
    const endpoint = isLogin ? '/auth/login' : '/auth/register';
    const body = isLogin ? { email, password } : { email, password, name };
    try {
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const data = await res.json();
      if (res.ok && data.status) {
        setToken(data.token);
        setUser(data.user);
        localStorage.setItem('token', data.token);
        localStorage.setItem('user', JSON.stringify(data.user));
      } else {
        alert(data.detail || 'Authentication failed');
      }
    } catch (err) {
      alert('Network error');
    }
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setFiles([]);
    setSessions([]);
    setSelectedSession(null);
    setMessages([]);
  };

  const fetchFiles = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/pdfs`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if(res.status === 401) return logout();
      const data = await res.json();
      if (data.status) {
        setFiles(data.data);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/api/pdfs`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData
      });
      const data = await res.json();
      if (res.ok && data.status) {
        await fetchFiles();
      } else {
        alert(data.detail || data.error || 'Upload failed');
      }
    } catch (err) {
      console.error(err);
      alert('Network or Server error during upload');
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  const deleteFile = async (fileId, e) => {
    e.stopPropagation();
    try {
      const res = await fetch(`${API_BASE}/api/pdfs/${fileId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        fetchFiles();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/users/${user._id}/sessions`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await res.json();
      if (data.status) {
        setSessions(data.data);
        if (!selectedSession && data.data.length > 0) {
          setSelectedSession(data.data[0]);
        }
      }
    } catch (err) {
      console.error(err);
    }
  };

  const createSession = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/users/${user._id}/sessions`, {
        method: 'POST',
        headers: { 
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json' 
        },
        body: JSON.stringify({ title: "New Chat" })
      });
      const data = await res.json();
      if (data.status) {
        await fetchSessions();
        setSelectedSession({ _id: data.data.session_id, title: data.data.title });
      }
    } catch (err) {
      console.error(err);
    }
  };

  const deleteSession = async (sessionId, e) => {
    e.stopPropagation();
    try {
      const res = await fetch(`${API_BASE}/api/users/${user._id}/sessions/${sessionId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        if (selectedSession?._id === sessionId) setSelectedSession(null);
        fetchSessions();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const fetchMessages = async () => {
    if (!user || !selectedSession) return;
    try {
      const res = await fetch(`${API_BASE}/api/users/${user._id}/sessions/${selectedSession._id}/messages`);
      const data = await res.json();
      if (data.status) {
        setMessages(data.data.reverse()); // Assuming backend returns newest first
      }
    } catch (err) {
      console.error(err);
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || !user || !selectedSession) return;

    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { type: 'Human', text: userMsg }]);
    setLoading(true);

    // Auto-rename session if it is still "New Chat"
    if (selectedSession.title === "New Chat" && messages.length === 0) {
      try {
        const titleRes = await fetch(`${API_BASE}/api/users/${user._id}/sessions/${selectedSession._id}/auto-title`, {
          method: 'PUT',
          headers: { 
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json' 
          },
          body: JSON.stringify({ message: userMsg })
        });
        const titleData = await titleRes.json();
        if (titleRes.ok && titleData.status) {
          setSelectedSession(prev => ({ ...prev, title: titleData.title }));
          fetchSessions();
        }
      } catch (e) {
        console.error("Failed to rename session", e);
      }
    }

    try {
      const res = await fetch(`${API_BASE}/api/users/${user._id}/sessions/${selectedSession._id}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: userMsg,
          lang: 'en',
          model: 'llama-3.1-8b-instant',
          debug: false
        })
      });
      const data = await res.json();
      if (res.ok && data.status) {
        setMessages(prev => [...prev, { type: 'AI', text: data.data.text }]);
      } else {
        const errorMsg = data.detail || data.error || 'Database or Server Error';
        setMessages(prev => [...prev, { type: 'Error', text: errorMsg }]);
      }
    } catch (err) {
      setMessages(prev => [...prev, { type: 'Error', text: 'Network Error - Backend might be unreachable' }]);
    } finally {
      setLoading(false);
    }
  };

  if (!token) {
    return (
      <div className="auth-container">
        <div className="auth-box glass">
          <h2>{isLogin ? 'Welcome Back' : 'Create Account'}</h2>
          <form onSubmit={handleAuth}>
            {!isLogin && (
              <input 
                type="text" 
                placeholder="Name" 
                value={name} 
                onChange={e => setName(e.target.value)} 
                required 
              />
            )}
            <input 
              type="email" 
              placeholder="Email" 
              value={email} 
              onChange={e => setEmail(e.target.value)} 
              required 
            />
            <input 
              type="password" 
              placeholder="Password" 
              value={password} 
              onChange={e => setPassword(e.target.value)} 
              required 
            />
            <button type="submit" className="btn-primary">
              {isLogin ? 'Login' : 'Sign Up'}
            </button>
          </form>
          <p onClick={() => setIsLogin(!isLogin)} className="toggle-auth">
            {isLogin ? "Don't have an account? Sign up" : "Already have an account? Login"}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="sidebar-section">
          <div className="sidebar-header">
            <h3>Chat Sessions</h3>
            <button className="new-chat-btn" onClick={createSession}>
              + New Chat
            </button>
          </div>
          <div className="session-list">
            {sessions.map(session => (
              <div 
                key={session._id} 
                className={`file-item ${selectedSession?._id === session._id ? 'active' : ''}`}
                onClick={() => setSelectedSession(session)}
              >
                <span className="file-name" title={session.title}>{session.title}</span>
                <button className="delete-btn" onClick={(e) => deleteSession(session._id, e)}>×</button>
              </div>
            ))}
            {sessions.length === 0 && <div className="empty-files">No active chats</div>}
          </div>
        </div>

        <div className="sidebar-section kb-section">
          <div className="sidebar-header">
            <h3>Knowledge Base</h3>
            <label className="upload-btn">
              {uploading ? 'Uploading...' : 'Upload PDF'}
              <input type="file" accept=".pdf" onChange={handleUpload} style={{ display: 'none' }} disabled={uploading}/>
            </label>
          </div>
          <div className="file-list">
            {files.map(file => (
              <div 
                key={file._id} 
                className="file-item"
              >
                <span className="file-name" title={file.filename}>{file.filename}</span>
                <button className="delete-btn" onClick={(e) => deleteFile(file._id, e)}>×</button>
              </div>
            ))}
            {files.length === 0 && <div className="empty-files">No PDFs uploaded</div>}
          </div>
        </div>
        
        <div className="sidebar-footer">
          <div className="user-info">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
            <span>{user?.name}</span>
          </div>
          <button className="logout-btn" onClick={logout}>
            Logout
          </button>
        </div>
      </aside>
      
      <main className="chat-container">
        {selectedSession ? (
          <>
            <header className="chat-header">
              <h2>Chatting in: <strong>{selectedSession.title}</strong></h2>
            </header>
            <div className="messages-area">
              {messages.length === 0 && <div className="empty-chat">Ask a question about your uploaded PDFs!</div>}
              {messages.map((msg, i) => (
                <div key={i} className={`message-wrapper ${msg.type === 'Human' ? 'human' : 'ai'}`}>
                  <div className={`message-bubble ${msg.type === 'Human' ? 'human-bubble' : 'ai-bubble'}`}>
                    {msg.type === 'Human' || msg.type === 'Error' ? (
                      msg.text
                    ) : (
                      <ReactMarkdown>{msg.text}</ReactMarkdown>
                    )}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="message-wrapper ai">
                  <div className="message-bubble loading">
                    <div className="dot-pulse"></div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
            <form className="chat-input-area" onSubmit={sendMessage}>
              <input 
                type="text" 
                placeholder="Ask something about the documents..." 
                value={input}
                onChange={e => setInput(e.target.value)}
                disabled={loading}
              />
              <button type="submit" disabled={!input.trim() || loading} className="send-btn">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>
              </button>
            </form>
          </>
        ) : (
          <div className="no-selection">
            <div className="logo-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
            </div>
            <h2>Welcome, {user?.name}!</h2>
            <p>Upload a PDF to your Knowledge Base, then start a chat to ask questions.</p>
            <button className="btn-primary" onClick={createSession}>+ New Chat</button>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
