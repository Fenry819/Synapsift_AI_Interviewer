"use client";

import React, { useState, useEffect } from 'react';
import { Upload, Briefcase, Play, Send, CheckCircle, AlertCircle, RefreshCw, Network, Lock, User, LogOut } from 'lucide-react';

type Step = 'AUTH' | 'SETUP' | 'INTERVIEW' | 'SUMMARY';

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

export default function SynapSiftScreener() {
  // Auth & Session States
  const [currentStep, setCurrentStep] = useState<Step>('AUTH');
  const [isLogin, setIsLogin] = useState<boolean>(true);
  const [authName, setAuthName] = useState<string>('');
  const [authEmail, setAuthEmail] = useState<string>('');
  const [authPassword, setAuthPassword] = useState<string>('');
  const [userToken, setUserToken] = useState<string | null>(null);
  const [userName, setUserName] = useState<string>('');

  // Interview States
  const [selectedRole, setSelectedRole] = useState('AI / Machine Learning Role');
  const [customRole, setCustomRole] = useState(''); 
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [inputAnswer, setInputAnswer] = useState<string>('');
  const [isAiThinking, setIsAiThinking] = useState<boolean>(false);
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);
  const [interviewId, setInterviewId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);

  const [analysisReport, setAnalysisReport] = useState<{ overallScore: number; summary: string; insights: string; breakdown: QAAnalysis[]; }>({
    overallScore: 0,
    summary: "",
    insights: "",
    breakdown: []
  });

  // Check for existing session token on load
  useEffect(() => {
    const token = localStorage.getItem("synapsift_token");
    const name = localStorage.getItem("synapsift_name");
    if (token && name) {
      setUserToken(token);
      setUserName(name);
      setCurrentStep('SETUP');
    }
  }, []);

  const handleLogOut = () => {
    localStorage.clear();
    setUserToken(null);
    setUserName('');
    setCurrentStep('AUTH');
  };

  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const endpoint = isLogin ? '/api/auth/login' : '/api/auth/signup';
    const payload = isLogin 
      ? { email: authEmail, password: authPassword }
      : { name: authName, email: authEmail, password: authPassword, role: 'candidate' };

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
        setUserToken(data.token);
        setUserName(data.name);
        setCurrentStep('SETUP');
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
    if (!uploadedFile || !userToken) return alert("Please upload a resume file to calibrate context.");
    if (selectedRole === 'Custom Role' && !customRole.trim()) return alert("Please enter a custom role.");
    
    // Core Bug Fix: Reset previous analytics states cleanly before starting
    setAnalysisReport({ overallScore: 0, summary: "Analyzing session data...", insights: "", breakdown: [] });
    setMessages([]); 
    setCurrentStep('INTERVIEW');
    setIsAiThinking(true);

    // Dynamic routing: use custom role string if "Custom Role" is selected
    const finalRole = selectedRole === 'Custom Role' ? customRole.trim() : selectedRole;

    const formData = new FormData();
    formData.append('role', finalRole);
    formData.append('resume', uploadedFile);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/interview/start', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${userToken}` },
        body: formData,
      });
      
      const data = await response.json();
      setInterviewId(data.interview_id);
      
      setMessages([{
        id: Date.now().toString(),
        sender: 'ai',
        text: data.first_question,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }]);
    } catch (error) {
      console.error("Error launching interview matrix pipeline:", error);
    } finally {
      setIsAiThinking(false);
    }
  };

  const handleSendAnswer = async () => {
    if (!inputAnswer.trim() || !interviewId || !userToken) return;

    const candidateMsg: Message = {
      id: Date.now().toString(),
      sender: 'candidate',
      text: inputAnswer,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, candidateMsg]);
    setInputAnswer('');
    setIsAiThinking(true);

    const finalRole = selectedRole === 'Custom Role' ? customRole.trim() : selectedRole;

    try {
      const response = await fetch('http://127.0.0.1:8000/api/interview/chat', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${userToken}`
        },
        body: JSON.stringify({
          interview_id: interviewId,
          message: candidateMsg.text,
          role: finalRole
        })
      });

      const data = await response.json();
      
      setMessages((prev) => [...prev, {
        id: (Date.now() + 1).toString(),
        sender: 'ai',
        text: data.reply,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }]);

      // Automated Termination State Engine
      if (data.status === 'COMPLETED') {
        setTimeout(() => handleTerminate(), 1500);
      }
    } catch (error) {
      console.error("Chat error:", error);
    } finally {
      setIsAiThinking(false);
    }
  };

  const handleTerminate = async () => {
    setCurrentStep('SUMMARY');
    setIsEvaluating(true); 
    if (!interviewId || !userToken) return;
    
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/interview/summary/${interviewId}`, {
        headers: { 'Authorization': `Bearer ${userToken}` }
      });
      const data = await response.json();
      setAnalysisReport(data);
    } catch (error) {
      console.error("Evaluation generation error:", error);
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
          <div className="flex items-center gap-4">
            <span className="text-xs text-slate-400">Candidate: <strong className="text-slate-200">{userName}</strong></span>
            <button onClick={handleLogOut} className="text-slate-400 hover:text-red-400 transition flex items-center gap-1.5 text-xs"><LogOut size={14} /> Log Out</button>
          </div>
        )}
      </header>

      <main className="flex-1 max-w-5xl w-full mx-auto p-6 flex flex-col justify-center">
        {currentStep === 'AUTH' && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 max-w-md w-full mx-auto shadow-2xl space-y-6">
            <div className="text-center space-y-1.5">
              <h2 className="text-2xl font-bold tracking-tight text-white">{isLogin ? "Welcome Back" : "Create Gateway Account"}</h2>
              <p className="text-xs text-slate-400">Enter fake credentials to initialize the gateway nodes.</p>
            </div>
            <form onSubmit={handleAuthSubmit} className="space-y-4">
              {!isLogin && (
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-slate-400 flex items-center gap-1.5"><User size={13} /> Full Name</label>
                  <input type="text" required value={authName} onChange={(e) => setAuthName(e.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500" placeholder="John Doe" />
                </div>
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
                {isLogin ? "Authenticate Credentials" : "Register Credentials"}
              </button>
            </form>
            <div className="text-center">
              <button onClick={() => setIsLogin(!isLogin)} className="text-xs text-indigo-400 hover:underline bg-transparent border-none">
                {isLogin ? "Need a candidate account? Register here" : "Already have credentials? Log in"}
              </button>
            </div>
          </div>
        )}

        {currentStep === 'SETUP' && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 max-w-xl w-full mx-auto shadow-2xl space-y-6">
            <div className="space-y-2">
              <h2 className="text-2xl font-bold tracking-tight text-white">Initialize Assessment</h2>
              <p className="text-sm text-slate-400">Upload your resume to calibrate our dynamic evaluation engine.</p>
            </div>
            
            {/* TARGET PROFILE DROPDOWN */}
            <div className="space-y-2">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2"><Briefcase size={14} className="text-indigo-400" /> Target Profile</label>
              <select 
                value={selectedRole} 
                onChange={(e) => {
                  setSelectedRole(e.target.value);
                  if (e.target.value !== 'Custom Role') setCustomRole('');
                }} 
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-slate-200 focus:outline-none focus:border-indigo-500 transition"
              >
                <option>AI / Machine Learning Role</option>
                <option>Data Science / Applied ML Role</option>
                <option>Advanced / Theoretical ML</option>
                <option>Backend Engineering Intern</option>
                <option>Custom Role</option>  
              </select>

              {/* CUSTOM ROLE TEXT INPUT */}
              {selectedRole === 'Custom Role' && (
                <input 
                  type="text" 
                  placeholder="e.g., Cloud Security Architect..." 
                  value={customRole} 
                  onChange={(e) => setCustomRole(e.target.value)} 
                  className="w-full bg-slate-950 border border-indigo-500/50 rounded-xl px-4 py-3 mt-3 text-slate-200 focus:outline-none focus:border-indigo-400 transition shadow-inner" 
                />
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
          <div className="bg-slate-900 border border-slate-800 rounded-2xl flex flex-col h-[75vh] shadow-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/80 flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-sm text-white">Active Stream Evaluation</h3>
                <p className="text-xs text-slate-400 truncate">Target: {selectedRole === 'Custom Role' ? customRole : selectedRole} | Matrix ID: <span className="font-mono">{interviewId || "Initializing..."}</span></p>
              </div>
              <button onClick={handleTerminate} className="text-xs bg-red-950 text-red-400 border border-red-900/50 px-3 py-1.5 rounded-lg hover:bg-red-900 hover:text-white transition">Terminate & Request Evaluation</button>
            </div>
            <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-slate-950/30">
              {messages.map((msg) => (
                <div key={msg.id} className={`flex flex-col ${msg.sender === 'ai' ? 'items-start' : 'items-end'}`}>
                  <div className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm shadow-md leading-relaxed whitespace-pre-wrap ${msg.sender === 'ai' ? 'bg-slate-900 border border-slate-800 text-slate-100 rounded-tl-none' : 'bg-indigo-600 text-white rounded-tr-none'}`}>
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
              <div className="flex items-center gap-3 bg-slate-950 border border-slate-800 rounded-xl px-4 py-2 focus-within:border-indigo-500 transition">
                <input type="text" placeholder="Formulate your technical response..." value={inputAnswer} onChange={(e) => setInputAnswer(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleSendAnswer()} className="flex-1 bg-transparent border-none text-sm text-slate-200 focus:outline-none py-2" />
                <button onClick={handleSendAnswer} className="p-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition"><Send size={14} /></button>
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
            
            {!isEvaluating && analysisReport.breakdown.length > 0 && (
              <div className="space-y-4">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Traceable Topic Log</h3>
                {analysisReport.breakdown.map((item, i) => (
                  <div key={i} className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
                    <div className="flex items-start justify-between gap-4">
                      <p className="text-sm font-medium text-slate-200">Q{i+1}: {item.question}</p>
                      <span className={`text-xs font-medium px-2 py-1 rounded-md ${item.score > 75 ? 'bg-emerald-500/10 text-emerald-400' : item.score > 50 ? 'bg-yellow-500/10 text-yellow-400' : 'bg-red-500/10 text-red-400'}`}>Score: {item.score}%</span>
                    </div>
                    <p className="text-sm text-slate-400 italic">"{item.answer || "[No response provided in transcript]"}"</p>
                    <div className="bg-slate-800/50 rounded-lg p-3 text-xs text-slate-300 flex items-start gap-2 border border-slate-700/50">
                      <AlertCircle size={14} className="mt-0.5 text-indigo-400 shrink-0" />
                      <p>{item.feedback}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {!isEvaluating && (
              <div className="flex justify-end">
                <button 
                  onClick={() => { setUploadedFile(null); setCurrentStep('SETUP'); }}
                  className="flex items-center gap-2 text-xs bg-slate-900 text-slate-300 border border-slate-800 px-4 py-2 rounded-xl hover:bg-slate-800 hover:text-white transition"
                >
                  <RefreshCw size={12} /> Start New Session
                </button>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}