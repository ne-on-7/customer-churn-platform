import type {
  PredictionResult,
  CustomerInput,
  ModelResults,
  DataOverview,
  ChurnRateData,
  BusinessImpactResult,
  ExperimentConfig,
  ExperimentSummary,
  PowerAnalysisResult,
  HighRiskCustomer,
  PredictionHistoryItem,
} from '../types';

const BASE = '/api';

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export async function predict(input: CustomerInput): Promise<PredictionResult> {
  return fetchJson('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });
}

export async function getModels(): Promise<ModelResults> {
  return fetchJson('/models');
}

export async function getHealth() {
  return fetchJson<{ status: string; models_loaded: number }>('/health');
}

export async function getDataOverview(): Promise<DataOverview> {
  return fetchJson('/data/overview');
}

export async function getChurnRates(groupBy: string, filters?: Record<string, unknown>): Promise<ChurnRateData[]> {
  const params = new URLSearchParams({ group_by: groupBy });
  if (filters) {
    Object.entries(filters).forEach(([k, v]) => {
      if (v !== undefined && v !== null) params.set(k, String(v));
    });
  }
  return fetchJson(`/data/churn-rates?${params}`);
}

export async function getDistribution(feature: string, filters?: Record<string, unknown>) {
  const params = new URLSearchParams({ feature });
  if (filters) {
    Object.entries(filters).forEach(([k, v]) => {
      if (v !== undefined && v !== null) params.set(k, String(v));
    });
  }
  return fetchJson<{ bins: number[]; churned: number[]; retained: number[] }>(`/data/distribution?${params}`);
}

export async function computeBusinessImpact(avgRevenue: number, retentionCost: number, monthsSaved: number): Promise<BusinessImpactResult> {
  return fetchJson('/business-impact', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ avg_monthly_revenue: avgRevenue, retention_cost: retentionCost, months_saved: monthsSaved }),
  });
}

export async function createExperiment(config: ExperimentConfig) {
  return fetchJson('/experiments', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
}

export async function listExperiments(): Promise<ExperimentSummary[]> {
  return fetchJson('/experiments');
}

export async function getExperiment(id: string) {
  return fetchJson<Record<string, unknown>>(`/experiments/${id}`);
}

export async function powerAnalysis(
  baselineChurnRate: number,
  mde: number,
  alpha: number,
  power: number
): Promise<PowerAnalysisResult> {
  return fetchJson('/experiments/power-analysis', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      baseline_churn_rate: baselineChurnRate,
      minimum_detectable_effect: mde,
      alpha,
      power,
      eligible_population: 7043,
    }),
  });
}

export async function getHighRiskCustomers(limit = 20): Promise<HighRiskCustomer[]> {
  return fetchJson(`/customers/high-risk?limit=${limit}`);
}

export async function predictBatch(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  return fetchJson<{ predictions: Record<string, unknown>[]; count: number }>('/predict/batch', {
    method: 'POST',
    body: formData,
  });
}

export async function getPredictionHistory(): Promise<PredictionHistoryItem[]> {
  return fetchJson('/predictions/history');
}

export function streamChat(
  messages: { role: string; content: string }[],
  onDelta: (data: { type: string; content?: string; tool?: string; result?: Record<string, unknown> }) => void,
  onDone: () => void,
  onError: (err: Error) => void,
) {
  const controller = new AbortController();

  fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) throw new Error(res.statusText);
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const raw = line.slice(6).trim();
            if (raw === '[DONE]') { onDone(); return; }
            try { onDelta(JSON.parse(raw)); } catch (e) { console.error('Failed to parse SSE chunk:', e); }
          }
        }
      }
      onDone();
    })
    .catch(onError);

  return () => controller.abort();
}
