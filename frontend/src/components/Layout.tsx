import { NavLink, Outlet } from 'react-router-dom'
import { useState, useEffect } from 'react'
import {
  MessageSquare,
  Target,
  BarChart3,
  Database,
  DollarSign,
  FlaskConical,
  ShieldAlert,
  History,
  Upload,
  Moon,
  Sun,
} from 'lucide-react'

const navItems = [
  { to: '/chat', icon: MessageSquare, label: 'AI Chat' },
  { to: '/predict', icon: Target, label: 'Predict' },
  { to: '/models', icon: BarChart3, label: 'Models' },
  { to: '/explorer', icon: Database, label: 'Explorer' },
  { to: '/impact', icon: DollarSign, label: 'Impact' },
  { to: '/experiments', icon: FlaskConical, label: 'A/B Testing' },
  { to: '/watchlist', icon: ShieldAlert, label: 'Watchlist' },
  { to: '/history', icon: History, label: 'History' },
  { to: '/batch', icon: Upload, label: 'Batch' },
]

export default function Layout() {
  const [dark, setDark] = useState(() => {
    const stored = localStorage.getItem('theme')
    return stored ? stored === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches
  })

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('theme', dark ? 'dark' : 'light')
  }, [dark])

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      {/* Sidebar */}
      <aside className="w-60 flex-shrink-0 bg-slate-900 dark:bg-slate-950 flex flex-col">
        <div className="px-5 py-5">
          <h1 className="text-lg font-semibold text-white tracking-tight">
            Churn Intelligence
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">ML Platform</p>
        </div>

        <nav className="flex-1 px-3 space-y-0.5 overflow-y-auto">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-indigo-500/15 text-indigo-400'
                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                }`
              }
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="p-3 border-t border-slate-800">
          <button
            onClick={() => setDark(!dark)}
            className="flex items-center gap-3 px-3 py-2.5 w-full rounded-lg text-sm text-slate-400 hover:text-white hover:bg-white/5 transition-colors"
          >
            {dark ? <Sun size={18} /> : <Moon size={18} />}
            {dark ? 'Light Mode' : 'Dark Mode'}
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  )
}
