"use client";

import React, { useState, useEffect, useRef } from 'react';
import { startReveal } from '../lib/reveal';
import { isPolicyClosure } from '../lib/interviewProgress';
import InterviewWorkspace, { type WorkspaceError } from '../components/InterviewWorkspace';
import AuthScreen from '../components/AuthScreen';
import SetupScreen from '../components/SetupScreen';
import ResultScreen, { type AnalysisReport } from '../components/ResultScreen';
import { initials } from '../lib/report';
import { RefreshCw, Network, LogOut, Users, FileText, Trash2, UserX } from 'lucide-react';

type Step = 'AUTH' | 'SETUP' | 'INTERVIEW' | 'SUMMARY' | 'ADMIN';

interface Message {
  id: string;
  sender: 'ai' | 'candidate';
  text: string;
  timestamp: string;
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
  const [authBusy, setAuthBusy] = useState<boolean>(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const [authNotice, setAuthNotice] = useState<string | null>(null);
  
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
  const [analysisReport, setAnalysisReport] = useState<AnalysisReport>({
    overallScore: 0, summary: "", insights: "", breakdown: []
  });


  // --- Interviewer reply reveal (presentation only: the backend has already validated the full text) ---
  const [revealing, setRevealing] = useState<{ id: string; shown: string } | null>(null);
  const cancelRevealRef = useRef<(() => void) | null>(null);
  const [uiError, setUiError] = useState<WorkspaceError | null>(null);   // inline error banner of the interview screen
  const busyRef = useRef<boolean>(false);   // true from the moment a request starts until the reply is fully revealed (blocks double submits)
  const epochRef = useRef<number>(0);       // bumped whenever the session is left; late responses from an old session are dropped
  const inputLocked = isAiThinking || revealing !== null;

  const cancelPending = () => {
    epochRef.current += 1;
    cancelRevealRef.current?.();
    cancelRevealRef.current = null;
    busyRef.current = false;
    setUiError(null);
    setRevealing(null);
    setIsAiThinking(false);
  };

  // Show a NEW interviewer message progressively. Messages already in the list are never animated.
  const appendInterviewerMessage = (text: string, onDone: () => void) => {
    const msg: Message = { id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, sender: 'ai', text, timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) };
    setMessages((prev) => [...prev, msg]);
    setIsAiThinking(false);
    const instant = typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    cancelRevealRef.current = startReveal(
      text,
      (shown) => setRevealing({ id: msg.id, shown }),
      () => {
        cancelRevealRef.current = null;
        setRevealing(null);
        busyRef.current = false;
        onDone();
      },
      { instant },
    );
  };

  useEffect(() => {
    return () => { epochRef.current += 1; cancelRevealRef.current?.(); };   // unmount: stop timers, ignore late responses
  }, []);

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
    setInputAnswer(e.target.value);     // height and focus are handled by the workspace's textarea
  };

  const handleLogOut = () => {
    cancelPending();
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

  // FastAPI errors carry `detail` as a string (HTTPException) or a list (validation errors).
  const getErrorMessage = (data: unknown, fallback: string): string => {
    const detail = (data as { detail?: unknown } | null)?.detail;
    return typeof detail === 'string' && detail ? detail : fallback;
  };

  const handleAbortSession = async () => {
    if (!confirm("WARNING: Are you sure you want to exit? This interview will NOT be saved or graded, and all progress will be permanently lost.")) return;
    try {
      // No interview was ever created (e.g. start failed): there is nothing to delete server-side.
      if (interviewId) {
        const response = await fetch(`http://127.0.0.1:8000/api/interview/abort/${interviewId}`, {
          method: 'DELETE',
          headers: { 'Authorization': `Bearer ${userToken}` }
        });
        if (!response.ok) {
          const data = await response.json().catch(() => null);
          return alert(getErrorMessage(data, "Could not abort the session. Please try again."));
        }
      }
      cancelPending();
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
    if (authBusy) return;
    const endpoint = isLogin ? '/api/auth/login' : '/api/auth/signup';
    const payload = isLogin
      ? { email: authEmail, password: authPassword }
      : { name: authName, email: authEmail, password: authPassword, role: authRole, admin_code: adminCode };

    setAuthBusy(true);
    setAuthError(null);
    setAuthNotice(null);
    try {
      const response = await fetch(`http://127.0.0.1:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await response.json().catch(() => null);
      if (!response.ok || !data) {
        setAuthError(getErrorMessage(data, "Authentication failed. Please check your details and try again."));
        return;
      }

      if (isLogin) {
        localStorage.setItem("synapsift_token", data.token);
        localStorage.setItem("synapsift_name", data.name);
        localStorage.setItem("synapsift_role", data.role);

        setUserToken(data.token);
        setUserName(data.name);
        setUserRole(data.role);
        setAuthPassword('');

        if (data.role === 'admin') {
          setCurrentStep('ADMIN');
          fetchCandidates(data.token);
        } else {
          setCurrentStep('SETUP');
        }
      } else {
        setAuthNotice("Account created. Please sign in.");
        setAuthPassword('');
        setIsLogin(true);
      }
    } catch (error) {
      console.error("Auth server connection error:", error);
      setAuthError("Could not reach the server. Please check your connection and try again.");
    } finally {
      setAuthBusy(false);
    }
  };

  const startSession = async () => {
    setIsInterviewComplete(false);
    if (!uploadedFile || !userToken) return alert("Please upload a resume file.");
    if (selectedRole === 'Custom Role' && !customRole.trim()) return alert("Please enter a custom role.");
    
    if (busyRef.current) return;
    cancelPending();
    busyRef.current = true;
    const epoch = epochRef.current;
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
      const data = await response.json().catch(() => null);
      if (epoch !== epochRef.current) return;     // the session was left while this request was running
      if (!response.ok || !data?.interview_id) {
        alert(getErrorMessage(data, "Could not start the interview. Please try again."));
        setInterviewId(null);
        setMessages([]);
        setCurrentStep('SETUP');
        busyRef.current = false;
        setIsAiThinking(false);
        return;
      }
      setInterviewId(data.interview_id);
      // The first question is revealed progressively too; the completion lock below applies once it is fully shown.
      appendInterviewerMessage(data.first_question, () => {
        if (data.status === 'COMPLETED') setIsInterviewComplete(true);
      });
    } catch (error) {
      if (epoch !== epochRef.current) return;
      console.error(error);
      // The server could not be reached: say so and go back to setup (same as a rejected start), instead of leaving an empty interview screen.
      alert("Could not reach the server to start the interview. Please check that the backend is running and try again.");
      setInterviewId(null);
      setMessages([]);
      setCurrentStep('SETUP');
      busyRef.current = false;
      setIsAiThinking(false);
    }
  };

  const handleSendAnswer = async () => {
    if (busyRef.current || !inputAnswer.trim() || !interviewId || !userToken) return;   // one request / reveal at a time
    busyRef.current = true;
    setUiError(null);
    const epoch = epochRef.current;

    const candidateMsg: Message = { id: Date.now().toString(), sender: 'candidate', text: inputAnswer, timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) };
    setMessages((prev) => [...prev, candidateMsg]);
    setInputAnswer('');
    
    setIsAiThinking(true);

    // The answer was not accepted: take it out of the chat, give it back to the user and unlock the input.
    const giveAnswerBack = (error: WorkspaceError) => {
      setUiError(error);
      setMessages((prev) => prev.filter((m) => m.id !== candidateMsg.id));
      setInputAnswer(candidateMsg.text);
      busyRef.current = false;
      setIsAiThinking(false);
    };

    try {
      const response = await fetch('http://127.0.0.1:8000/api/interview/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${userToken}` },
        body: JSON.stringify({ interview_id: interviewId, message: candidateMsg.text, role: selectedRole === 'Custom Role' ? customRole.trim() : selectedRole })
      });
      const data = await response.json().catch(() => null);
      if (epoch !== epochRef.current) return;     // the session was left while this request was running
      if (!response.ok || typeof data?.reply !== 'string') {
        giveAnswerBack({
          title: response.status === 503 ? "Interviewer temporarily unavailable" : "Your answer was not sent",
          message: getErrorMessage(data, "Your answer could not be sent. Please try again."),
        });
        return;
      }

      // Show the approved reply progressively; the completion check (submit button) runs once it is fully visible.
      appendInterviewerMessage(data.reply, () => {
        if (data.status === 'COMPLETED') setIsInterviewComplete(true);
      });
    } catch (error) {
      if (epoch !== epochRef.current) return;
      console.error(error);
      giveAnswerBack({ title: "Server unavailable", message: "Could not reach the server. Your answer was not sent; it has been restored so you can try again." });
    }
  };

  const handleTerminate = async (specificId?: string) => {
    const idToFetch = specificId || interviewId;
    if (!idToFetch || !userToken) return;
    
    setUiError(null);
    setCurrentStep('SUMMARY');
    setIsEvaluating(true); 
    
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/interview/summary/${idToFetch}`, {
        headers: { 'Authorization': `Bearer ${userToken}` }
      });
      const data = await response.json().catch(() => null);
      if (!response.ok || !data) {
        // Admin opened this from the dashboard; a candidate retries from the finished chat screen.
        const detail = getErrorMessage(data, "Could not load the evaluation. Please try again.");
        if (specificId) alert(detail); else setUiError({ title: "Evaluation temporarily unavailable", message: detail });
        setCurrentStep(specificId ? 'ADMIN' : 'INTERVIEW');
        return;
      }
      setAnalysisReport({ ...data, breakdown: Array.isArray(data.breakdown) ? data.breakdown : [] });
    } catch (error) {
      console.error(error);
      const detail = "Could not reach the server. Please try again.";
      if (specificId) alert(detail); else setUiError({ title: "Server unavailable", message: detail });
      setCurrentStep(specificId ? 'ADMIN' : 'INTERVIEW');
    } finally {
      setIsEvaluating(false); 
    }
  };

  return (
    <div className="sf-app font-sans">
      <header className="sticky top-0 z-30 border-b border-white/[0.06] bg-[rgba(7,9,18,0.72)] backdrop-blur-md">
        <div className="mx-auto flex h-16 w-full max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-500 text-white shadow-[0_8px_24px_-8px_rgba(99,102,241,0.9),inset_0_1px_0_rgba(255,255,255,0.25)]">
              <Network size={18} />
            </span>
            <span className="text-[17px] font-semibold tracking-tight text-slate-100">SynapSift <span className="font-normal text-indigo-300">AI</span></span>
          </div>

          {userToken && (
            <div className="sf-enter-soft flex items-center gap-2 sm:gap-3">
              <span className="flex items-center gap-2.5 rounded-full border border-white/[0.08] bg-white/[0.03] py-1 pl-1 pr-3">
                <span aria-hidden className="flex h-7 w-7 items-center justify-center rounded-full bg-indigo-400/20 text-[11px] font-semibold text-indigo-100">{initials(userName)}</span>
                <span className="hidden text-[12.5px] text-slate-200 sm:inline">{userName}</span>
                <span className="hidden text-[11px] text-slate-500 md:inline">{userRole === 'admin' ? 'Administrator' : 'Candidate'}</span>
              </span>
              <button onClick={handleLogOut} className="sf-btn sf-btn-ghost px-3 py-1.5 text-[12.5px]" aria-label="Log out"><LogOut size={14} /> <span className="hidden sm:inline">Log out</span></button>
              <button onClick={handleDeleteAccount} className="sf-focus rounded-lg p-2 text-slate-500 transition-colors duration-200 hover:bg-rose-500/10 hover:text-rose-300" aria-label="Delete account" title="Delete account"><UserX size={15} /></button>
            </div>
          )}
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col justify-center px-4 py-6 sm:px-6">
        {currentStep === 'AUTH' && (
          <AuthScreen
            isLogin={isLogin} setIsLogin={setIsLogin}
            authName={authName} setAuthName={setAuthName}
            authEmail={authEmail} setAuthEmail={setAuthEmail}
            authPassword={authPassword} setAuthPassword={setAuthPassword}
            authRole={authRole} setAuthRole={setAuthRole}
            adminCode={adminCode} setAdminCode={setAdminCode}
            busy={authBusy} error={authError} notice={authNotice}
            onDismiss={() => { setAuthError(null); setAuthNotice(null); }}
            onSubmit={handleAuthSubmit}
          />
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
          <SetupScreen
            userName={userName}
            selectedRole={selectedRole} setSelectedRole={setSelectedRole}
            customRole={customRole} setCustomRole={setCustomRole}
            file={uploadedFile} setFile={setUploadedFile}
            onStart={startSession}
          />
        )}

        {currentStep === 'INTERVIEW' && (
          <InterviewWorkspace
            role={selectedRole === 'Custom Role' ? customRole : selectedRole}
            messages={messages}
            revealing={revealing}
            isAiThinking={isAiThinking}
            inputLocked={inputLocked}
            isComplete={isInterviewComplete}
            policyClosed={isPolicyClosure(messages[messages.length - 1]?.text)}
            inputAnswer={inputAnswer}
            error={uiError}
            onInputChange={handleInputResize}
            onSend={handleSendAnswer}
            onAbort={handleAbortSession}
            onViewEvaluation={() => handleTerminate()}
            onReturnToSetup={() => { cancelPending(); setUploadedFile(null); setInterviewId(null); setCurrentStep('SETUP'); }}
            onDismissError={() => setUiError(null)}
          />
        )}

        {currentStep === 'SUMMARY' && (
          <ResultScreen
            report={analysisReport}
            isEvaluating={isEvaluating}
            isAdmin={userRole === 'admin'}
            showResume={showResume}
            onToggleResume={() => setShowResume(!showResume)}
            onBackToDashboard={() => setCurrentStep('ADMIN')}
            onNewSession={() => { setUploadedFile(null); setCurrentStep('SETUP'); }}
          />
        )}
      </main>
    </div>
  );
}