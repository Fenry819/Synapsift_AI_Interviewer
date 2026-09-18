"use client";

import React, { useState, useEffect, useRef } from 'react';
import { Upload, Briefcase, Play, Send, CheckCircle, AlertCircle, RefreshCw, Network, Lock, User, LogOut, Users, FileText, Trash2, UserX, FileBadge } from 'lucide-react';

type Step = 'AUTH' | 'SETUP' | 'INTERVIEW' | 'SUMMARY' | 'ADMIN';

interface Message {
  id: string;
  sender: 'ai' | 'candidate';
  text: string;
  timestamp: string;
}

interface QAAnalysis {
  question: string;
  answer: string;
  score: number;
  feedback: string;
}

interface CandidateRecord {
  name: string;
  email: string;
  interview_id: string;
  role: string;
  status: string;
  date: string;
}

export default function SynapSiftScreener() {
  const [isInterviewComplete, setIsInterviewComplete] = useState<boolean>(false);
  const [currentStep, setCurrentStep] = useState<Step>('AUTH');
  const [isLogin, setIsLogin] = useState<boolean>(true);
  const [authName, setAuthName] = useState<string>('');
  const [authEmail, setAuthEmail] = useState<string>('');
  const [authPassword, setAuthPassword] = useState<string>('');
  const [authRole, setAuthRole] = useState<string>('candidate');
  const [adminCode, setAdminCode] = useState<string>('');
  
  const [userToken, setUserToken] = useState<string | null>(null);
  const [userName, setUserName] = useState<string>('');
  const [userRole, setUserRole] = useState<string>('');

  const [selectedRole, setSelectedRole] = useState('AI / Machine Learning Role');
  const [customRole, setCustomRole] = useState(''); 
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [inputAnswer, setInputAnswer] = useState<string>('');
  const [isAiThinking, setIsAiThinking] = useState<boolean>(false);
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);
  const [interviewId, setInterviewId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [showResume, setShowResume] = useState<boolean>(false);

  const [candidates, setCandidates] = useState<CandidateRecord[]>([]);
  const [analysisReport, setAnalysisReport] = useState<{ overallScore: number; summary: string; insights: string; resume_url?: string; breakdown: QAAnalysis[]; }>({
    overallScore: 0, summary: "", insights: "", breakdown: []
  });

  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const token = localStorage.getItem("synapsift_token");
    const name = localStorage.getItem("synapsift_name");
    const role = localStorage.getItem("synapsift_role");
    
    if (token && name && role) {
      setUserToken(token);
      setUserName(name);
      setUserRole(role);
      if (role === 'admin') {
        setCurrentStep('ADMIN');
        fetchCandidates(token);
      } else {
        setCurrentStep('SETUP');
      }
    }
  }, []);

  const handleInputResize = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputAnswer(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${e.target.scrollHeight}px`;
  };

  const handleLogOut = () => {
    localStorage.clear();
    setUserToken(null);
    setUserName('');
    setUserRole('');
    setCurrentStep('AUTH');
  };

  const handleDeleteAccount = async () => {
    if (!confirm("Are you sure you want to permanently delete your account and all history? This cannot be undone.")) return;
    try {
      const response = await fetch('http://127.0.0.1:8000/api/user/account', {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${userToken}` }
      });
      if (response.ok) handleLogOut();
    } catch (error) {
      console.error("Failed to delete account", error);
    }
  };

  const handleAdminDeleteInterview = async (id: string) => {
    if (!confirm("Delete this candidate's interview record permanently?")) return;
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/admin/interview/${id}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${userToken}` }
      });
      if (response.ok) fetchCandidates(userToken!);
    } catch (error) {
      console.error("Failed to delete interview", error);
    }
  };

  const handleAbortSession = async () => {
    if (!confirm("WARNING: Are you sure you want to exit? This interview will NOT be saved or graded, and all progress will be permanently lost.")) return;
    try {
      await fetch(`http://127.0.0.1:8000/api/interview/abort/${interviewId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${userToken}` }
      });
      setUploadedFile(null);
      setMessages([]);
      setInterviewId(null);
      setCurrentStep('SETUP');
    } catch (error) {
      console.error("Failed to abort session", error);
    }
  };

  const fetchCandidates = async (token: string) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/admin/candidates', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) setCandidates(await response.json());
    } catch (error) {
      console.error("Failed to fetch admin data", error);
    }
  };

  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const endpoint = isLogin ? '/api/auth/login' : '/api/auth/signup';
    const payload = isLogin 
      ? { email: authEmail, password: authPassword }
      : { name: authName, email: authEmail, password: authPassword, role: authRole, admin_code: adminCode };

    try {
      const response = await fetch(`http://127.0.0.1:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) return alert(data.detail || "Authentication failed");

      if (isLogin) {
        localStorage.setItem("synapsift_token", data.token);
        localStorage.setItem("synapsift_name", data.name);
        localStorage.setItem("synapsift_role", data.role);
        
        setUserToken(data.token);
        setUserName(data.name);
        setUserRole(data.role);
        
        if (data.role === 'admin') {
          setCurrentStep('ADMIN');
          fetchCandidates(data.token);
        } else {
          setCurrentStep('SETUP');
        }
      } else {
        alert("Registration complete! Please log in.");
        setIsLogin(true);
      }
    } catch (error) {
      console.error("Auth server connection error:", error);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) setUploadedFile(e.target.files[0]);
  };

  const startSession = async () => {
    setIsInterviewComplete(false);
    if (!uploadedFile || !userToken) return alert("Please upload a resume file.");
    if (selectedRole === 'Custom Role' && !customRole.trim()) return alert("Please enter a custom role.");
    
    setAnalysisReport({ overallScore: 0, summary: "Analyzing session data...", insights: "", breakdown: [] });
    setMessages([]); 
    setCurrentStep('INTERVIEW');
    setIsAiThinking(true);

    const formData = new FormData();
    formData.append('role', selectedRole === 'Custom Role' ? customRole.trim() : selectedRole);
    formData.append('resume', uploadedFile);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/interview/start', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${userToken}` },
        body: formData,
      });
      const data = await response.json();
      setInterviewId(data.interview_id);
      setMessages([{ id: Date.now().toString(), sender: 'ai', text: data.first_question, timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }]);
      
      // === 🛑 THE UI LOCK GUARDRAIL ===
      // If the backend instantly rejects the role, lock the chat box!
      if (data.status === 'COMPLETED') {
        setIsInterviewComplete(true); 
      }

    } catch (error) {
      console.error(error);
    } finally {
      setIsAiThinking(false);
    }
  };

  const handleSendAnswer = async () => {
    if (!inputAnswer.trim() || !interviewId || !userToken) return;

    const candidateMsg: Message = { id: Date.now().toString(), sender: 'candidate', text: inputAnswer, timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) };
    setMessages((prev) => [...prev, candidateMsg]);
    setInputAnswer('');
    
    if (textareaRef.current) {
      textareaRef.current.style.height = '44px'; // Reset height
    }
    
    setIsAiThinking(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/interview/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${userToken}` },
        body: JSON.stringify({ interview_id: interviewId, message: candidateMsg.text, role: selectedRole === 'Custom Role' ? customRole.trim() : selectedRole })
      });
      const data = await response.json();
      
      // Add the AI's message to the chat
      setMessages((prev) => [...prev, { id: (Date.now() + 1).toString(), sender: 'ai', text: data.reply, timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }]);

      // Check the exact status from the backend to reveal the submit button!
      if (data.status === 'COMPLETED') {
        setIsInterviewComplete(true); 
      }
    } catch (error) {
      console.error(error);
    } finally {
      setIsAiThinking(false);
    }
  };

  const handleTerminate = async (specificId?: string) => {
    const idToFetch = specificId || interviewId;
    if (!idToFetch || !userToken) return;
    
    setCurrentStep('SUMMARY');
    setIsEvaluating(true); 
    
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/interview/summary/${idToFetch}`, {
        headers: { 'Authorization': `Bearer ${userToken}` }
      });
      setAnalysisReport(await response.json());
    } catch (error) {
      console.error(error);
    } finally {
      setIsEvaluating(false); 
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white shadow-lg shadow-indigo-500/20">
            <Network size={18} />
          </div>
          <h1 className="font-semibold text-lg tracking-wide text-slate-200">SynapSift <span className="text-indigo-400 font-normal">AI</span></h1>
        </div>
        
        {userToken && (
          <div className="flex items-center gap-5">
            <span className="text-xs text-slate-400">
              {userRole === 'admin' ? 'Admin:' : 'Candidate:'} <strong className="text-slate-200">{userName}</strong>
            </span>
            <div className="flex items-center gap-3 border-l border-slate-700 pl-4">
              <button onClick={handleDeleteAccount} className="text-slate-400 hover:text-red-400 transition flex items-center gap-1.5 text-xs"><UserX size={14} /> Delete Account</button>
              <button onClick={handleLogOut} className="text-slate-400 hover:text-white transition flex items-center gap-1.5 text-xs"><LogOut size={14} /> Log Out</button>
            </div>
          </div>
        )}
      </header>

      <main className="flex-1 max-w-5xl w-full mx-auto p-6 flex flex-col justify-center">
        {currentStep === 'AUTH' && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 max-w-md w-full mx-auto shadow-2xl space-y-6">
            <div className="text-center space-y-1.5">
              <h2 className="text-2xl font-bold tracking-tight text-white">{isLogin ? "System Login" : "Create Account"}</h2>
              <p className="text-xs text-slate-400">Authenticate to access the SynapSift platform.</p>
            </div>
            <form onSubmit={handleAuthSubmit} className="space-y-4">
              {!isLogin && (
                <>
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-slate-400 flex items-center gap-1.5"><User size={13} /> Full Name</label>
                    <input type="text" required value={authName} onChange={(e) => setAuthName(e.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500" placeholder="John Doe" />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-slate-400 flex items-center gap-1.5"><Briefcase size={13} /> Account Type</label>
                    <select value={authRole} onChange={(e) => { setAuthRole(e.target.value); setAdminCode(''); }} className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500">
                      <option value="candidate">Candidate (Take Interview)</option>
                      <option value="admin">Administrator (Review Results)</option>
                    </select>
                  </div>
                  {authRole === 'admin' && (
                    <div className="space-y-1.5">
                      <label className="text-xs font-medium text-indigo-400 flex items-center gap-1.5"><Lock size={13} /> Admin Access Code</label>
                      <input type="text" required value={adminCode} onChange={(e) => setAdminCode(e.target.value)} className="w-full bg-slate-950 border border-indigo-500/50 rounded-xl px-4 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-400 shadow-inner" placeholder="Enter recruiter code..." />
                    </div>
                  )}
                </>
              )}
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-slate-400 flex items-center gap-1.5">@ Virtual Email</label>
                <input type="email" required value={authEmail} onChange={(e) => setAuthEmail(e.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500" placeholder="your@email.com" />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-slate-400 flex items-center gap-1.5"><Lock size={13} /> Secure Keycode</label>
                <input type="password" required value={authPassword} onChange={(e) => setAuthPassword(e.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500" placeholder="••••••••" />
              </div>
              <button type="submit" className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white font-medium rounded-xl text-sm transition shadow-lg shadow-indigo-600/25 mt-2">
                {isLogin ? "Authenticate Credentials" : "Register Account"}
              </button>
            </form>
            <div className="text-center">
              <button onClick={() => setIsLogin(!isLogin)} className="text-xs text-indigo-400 hover:underline bg-transparent border-none">
                {isLogin ? "Need an account? Register here" : "Already have an account? Log in"}
              </button>
            </div>
          </div>
        )}

        {currentStep === 'ADMIN' && (
          <div className="space-y-6 w-full max-w-5xl mx-auto py-4">
             <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-white flex items-center gap-2"><Users className="text-indigo-400" /> Candidate Pipeline</h2>
                  <p className="text-sm text-slate-400 mt-1">Review completed and ongoing technical assessments.</p>
                </div>
                <button onClick={() => fetchCandidates(userToken!)} className="flex items-center gap-2 text-xs bg-slate-900 border border-slate-800 px-4 py-2 rounded-xl hover:bg-slate-800 transition">
                  <RefreshCw size={14} /> Refresh Data
                </button>
             </div>
             
             <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
                <table className="w-full text-left text-sm text-slate-300">
                  <thead className="bg-slate-950/50 text-xs uppercase text-slate-500 border-b border-slate-800">
                    <tr>
                      <th className="px-6 py-4 font-medium">Candidate Name</th>
                      <th className="px-6 py-4 font-medium">Email</th>
                      <th className="px-6 py-4 font-medium">Target Role</th>
                      <th className="px-6 py-4 font-medium">Status</th>
                      <th className="px-6 py-4 font-medium text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50">
                    {candidates.length === 0 ? (
                      <tr><td colSpan={5} className="px-6 py-8 text-center text-slate-500">No candidate records found.</td></tr>
                    ) : (
                      candidates.map((c) => (
                        <tr key={c.interview_id} className="hover:bg-slate-800/20 transition group">
                          <td className="px-6 py-4 font-medium text-white">{c.name}</td>
                          <td className="px-6 py-4 text-slate-400">{c.email}</td>
                          <td className="px-6 py-4 text-indigo-300">{c.role}</td>
                          <td className="px-6 py-4">
                            <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${c.status === 'COMPLETED' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-yellow-500/10 text-yellow-400'}`}>
                              {c.status}
                            </span>
                          </td>
                          <td className="px-6 py-4 flex items-center justify-end">
                            {c.status === 'COMPLETED' ? (
                              <button onClick={() => handleTerminate(c.interview_id)} className="text-xs bg-indigo-600 hover:bg-indigo-500 text-white px-3 py-1.5 rounded-lg transition flex items-center gap-1.5">
                                <FileText size={12} /> View Report
                              </button>
                            ) : (
                              <span className="text-xs text-slate-500 px-3">In Progress</span>
                            )}
                            <button onClick={() => handleAdminDeleteInterview(c.interview_id)} className="text-xs bg-red-950 hover:bg-red-900 text-red-400 px-3 py-1.5 rounded-lg transition flex items-center gap-1.5 ml-2 opacity-0 group-hover:opacity-100" title="Reject / Delete Record">
                              <Trash2 size={14} />
                            </button>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
             </div>
          </div>
        )}

        {currentStep === 'SETUP' && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 max-w-xl w-full mx-auto shadow-2xl space-y-6">
            <div className="space-y-2">
              <h2 className="text-2xl font-bold tracking-tight text-white">Initialize Assessment</h2>
              <p className="text-sm text-slate-400">Upload your latest resume to calibrate our dynamic evaluation engine.</p>
            </div>
            
            <div className="space-y-2">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2"><Briefcase size={14} className="text-indigo-400" /> Target Profile</label>
              <select value={selectedRole} onChange={(e) => { setSelectedRole(e.target.value); if (e.target.value !== 'Custom Role') setCustomRole(''); }} className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-200 focus:outline-none focus:border-indigo-500 transition">
                <option>AI / Machine Learning Role</option>
                <option>Data Science / Applied ML Role</option>
                <option>Advanced / Theoretical ML</option>
                <option>Backend Engineering Intern</option>
                <option>Custom Role</option>  
              </select>
              {selectedRole === 'Custom Role' && (
                <input type="text" placeholder="e.g., Cloud Security Architect..." value={customRole} onChange={(e) => setCustomRole(e.target.value)} className="w-full bg-slate-950 border border-indigo-500/50 rounded-xl px-4 py-3 mt-3 text-slate-200 focus:outline-none focus:border-indigo-400 transition shadow-inner" />
              )}
            </div>

            <div className="space-y-2">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2"><Upload size={14} className="text-indigo-400" /> Professional Resume</label>
              <div className="border-2 border-dashed border-slate-800 hover:border-indigo-500/50 rounded-xl p-6 bg-slate-950/50 flex flex-col items-center justify-center gap-3 relative cursor-pointer group">
                <input type="file" accept=".pdf" onChange={handleFileChange} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" />
                <div className="h-12 w-12 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center group-hover:scale-105 transition text-indigo-400"><Upload size={20} /></div>
                <p className="text-sm font-medium text-slate-300">{uploadedFile ? uploadedFile.name : "Click or drag resume file here"}</p>
              </div>
            </div>
            <button onClick={startSession} disabled={!uploadedFile} className={`w-full py-3.5 px-4 rounded-xl font-medium flex items-center justify-center gap-2 transition ${uploadedFile ? 'bg-indigo-600 hover:bg-indigo-500 text-white cursor-pointer shadow-lg shadow-indigo-600/20' : 'bg-slate-800 text-slate-500 cursor-not-allowed'}`}>
              <Play size={16} fill="currentColor" /> Compile Pipeline & Start
            </button>
          </div>
        )}

        {currentStep === 'INTERVIEW' && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl flex flex-col h-[80vh] shadow-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/80 flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-sm text-white">Active Stream Evaluation</h3>
                <p className="text-xs text-slate-400 truncate">Target: {selectedRole === 'Custom Role' ? customRole : selectedRole} | Matrix ID: <span className="font-mono">{interviewId || "Initializing..."}</span></p>
              </div>
              <button onClick={handleAbortSession} className="text-xs bg-slate-950 text-slate-400 border border-slate-800 px-3 py-1.5 rounded-lg hover:bg-red-950 hover:text-red-400 transition">
                Abort Session
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-slate-950/30">
              {messages.map((msg) => (
                <div key={msg.id} className={`flex flex-col ${msg.sender === 'ai' ? 'items-start' : 'items-end'}`}>
                  <div className={`max-w-[85%] rounded-2xl px-5 py-4 shadow-md whitespace-pre-wrap ${msg.sender === 'ai' ? 'bg-slate-900 border border-slate-800 text-slate-100 rounded-tl-none text-[15px] leading-relaxed font-normal tracking-wide' : 'bg-indigo-600 text-white rounded-tr-none text-[14px] leading-relaxed'}`}>
                    {msg.text}
                  </div>
                </div>
              ))}
              {isAiThinking && (
                <div className="flex items-center gap-2.5 text-xs text-slate-500 font-mono italic pl-2 bg-slate-900/20 py-2 w-max rounded-lg">
                  <RefreshCw size={12} className="animate-spin text-indigo-400" /> Synchronizing language pipeline models...
                </div>
              )}
            </div>
            
            <div className="p-4 border-t border-slate-800 bg-slate-900">
              <div className="flex flex-col gap-3">
                
                {!isInterviewComplete ? (
                  <>
                    <div className="flex items-end gap-3 bg-slate-950 border border-slate-800 rounded-xl px-4 py-2 focus-within:border-indigo-500 transition">
                      <textarea 
                        ref={textareaRef}
                        placeholder="Formulate your technical response..." 
                        value={inputAnswer} 
                        onChange={handleInputResize}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSendAnswer();
                          }
                        }} 
                        className="flex-1 bg-transparent border-none text-sm text-slate-200 focus:outline-none py-2 resize-none min-h-[40px] max-h-32 overflow-y-auto custom-scrollbar" 
                        rows={1}
                      />
                      <button onClick={handleSendAnswer} className="p-2 mb-1 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition"><Send size={14} /></button>
                    </div>
                    <div className="flex justify-between items-center px-1">
                      <span className="text-[11px] text-slate-500">Press <strong className="text-slate-400">Enter</strong> to send, <strong className="text-slate-400">Shift + Enter</strong> for new line.</span>
                    </div>
                  </>
                ) : (
                  <div className="flex flex-col items-center justify-center p-4 bg-slate-950 border border-slate-800 rounded-xl gap-3">
                    {messages.length > 0 && (
                      messages[messages.length - 1].text.includes("unable to conduct") || 
                      messages[messages.length - 1].text.includes("session is closed") ||
                      messages[messages.length - 1].text.includes("unwilling to proceed")
                    ) ? (
                      <>
                        <p className="text-sm text-red-400 font-medium">Session closed due to domain constraints or policy violation.</p>
                        <button onClick={() => { setUploadedFile(null); setInterviewId(null); setCurrentStep('SETUP'); }} className="text-sm font-medium bg-slate-800 hover:bg-slate-700 text-white px-6 py-2.5 rounded-lg transition flex items-center gap-2 border border-slate-700">
                          <RefreshCw size={16} /> Return to Setup
                        </button>
                      </>
                    ) : (
                      <>
                        <p className="text-sm text-slate-400">The technical assessment has been concluded.</p>
                        <button onClick={() => handleTerminate()} className="text-sm font-medium bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-500/20 px-6 py-2.5 rounded-lg transition flex items-center gap-2">
                          <CheckCircle size={16} /> End Interview & Get Result
                        </button>
                      </>
                    )}
                  </div>
                )}
                
              </div>
            </div>
          </div>
        )}

        {currentStep === 'SUMMARY' && (
          <div className="space-y-6 w-full max-w-4xl mx-auto py-4">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex items-center gap-6 shadow-xl">
              <div className="relative h-28 w-28 flex items-center justify-center rounded-full border-4 border-slate-800 bg-slate-950 shrink-0">
                {isEvaluating ? (
                  <RefreshCw size={24} className="animate-spin text-indigo-400" />
                ) : (
                  <span className="text-3xl font-black text-white">{analysisReport.overallScore}<span className="text-sm font-normal text-slate-500">/100</span></span>
                )}
              </div>
              <div className="flex-1 space-y-1.5">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <CheckCircle className={isEvaluating ? "text-slate-500 animate-pulse" : "text-emerald-500"} size={20} /> 
                  {isEvaluating ? "Running Deep Evaluation Diagnostics..." : "Evaluation Complete"}
                </h2>
                <p className="text-sm text-slate-400 leading-relaxed">{analysisReport.summary}</p>
                {analysisReport.insights && (
                  <div className="mt-4 p-3 bg-indigo-900/20 border border-indigo-500/20 rounded-lg">
                    <p className="text-sm text-indigo-300"><strong className="font-semibold text-indigo-200">Basic Insights:</strong> {analysisReport.insights}</p>
                  </div>
                )}
              </div>
            </div>

            {/* Update your analysisReport state interface to expect resume_url instead of resume */}
            {!isEvaluating && userRole === 'admin' && analysisReport.resume_url && (
               <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
                  <button onClick={() => setShowResume(!showResume)} className="flex items-center justify-between w-full text-left">
                    <span className="text-sm font-semibold text-indigo-300 flex items-center gap-2"><FileBadge size={16} /> Candidate Resume (Original PDF)</span>
                    <span className="text-xs text-slate-500">{showResume ? 'Hide' : 'View'}</span>
                  </button>
                  {showResume && (
                    <div className="mt-4 rounded-lg overflow-hidden border border-slate-800 h-[600px]">
                      <iframe 
                        src={`${analysisReport.resume_url}#toolbar=0`} 
                        width="100%" 
                        height="100%" 
                        title="Candidate Resume" 
                        className="border-none bg-slate-950" 
                      />
                    </div>
                  )}
               </div>
            )}
            
            {!isEvaluating && analysisReport.breakdown.length > 0 && (
              <div className="space-y-4">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Traceable Topic Log (Top Questions)</h3>
                {analysisReport.breakdown.map((item, i) => (
                  <div key={i} className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
                    <div className="flex items-start justify-between gap-4">
                      <p className="text-sm font-medium text-slate-200">Q: {item.question}</p>
                      <span className={`text-xs font-medium px-2 py-1 rounded-md shrink-0 ${item.score > 75 ? 'bg-emerald-500/10 text-emerald-400' : item.score > 50 ? 'bg-yellow-500/10 text-yellow-400' : 'bg-red-500/10 text-red-400'}`}>Score: {item.score}%</span>
                    </div>
                    <p className="text-[13px] text-slate-400 italic bg-slate-950 p-3 rounded-lg border border-slate-800/50">"{item.answer || "[No response provided]"}"</p>
                    <div className="bg-slate-800/30 rounded-lg p-3 text-xs text-slate-300 flex items-start gap-2 border border-slate-700/30">
                      <AlertCircle size={14} className="mt-0.5 text-indigo-400 shrink-0" />
                      <p>{item.feedback}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {!isEvaluating && (
              <div className="flex justify-end gap-3 pb-8">
                {userRole === 'admin' ? (
                   <button onClick={() => setCurrentStep('ADMIN')} className="flex items-center gap-2 text-xs bg-indigo-600 text-white px-4 py-2 rounded-xl hover:bg-indigo-500 transition">Back to Dashboard</button>
                ) : (
                  <button onClick={() => { setUploadedFile(null); setCurrentStep('SETUP'); }} className="flex items-center gap-2 text-xs bg-slate-900 text-slate-300 border border-slate-800 px-4 py-2 rounded-xl hover:bg-slate-800 hover:text-white transition">
                    <RefreshCw size={12} /> Start New Session
                  </button>
                )}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}