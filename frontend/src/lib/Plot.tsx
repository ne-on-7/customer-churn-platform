import plotlyFactory from 'react-plotly.js/factory';
import PlotlyLib from 'plotly.js-dist-min';

// CJS/ESM interop: default import may be the namespace object or the value
const createPlotlyComponent: any = (plotlyFactory as any).default ?? plotlyFactory;
const Plotly: any = (PlotlyLib as any).default ?? PlotlyLib;

const Plot = createPlotlyComponent(Plotly);
export default Plot;
