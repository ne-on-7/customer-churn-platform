declare module 'react-plotly.js/factory' {
  import { ComponentType } from 'react';
  function createPlotlyComponent(plotly: object): ComponentType<{
    data: Plotly.Data[];
    layout?: Partial<Plotly.Layout>;
    config?: Partial<Plotly.Config>;
    className?: string;
    style?: React.CSSProperties;
    [key: string]: unknown;
  }>;
  export default createPlotlyComponent;
}

declare module 'plotly.js/dist/plotly' {
  const Plotly: object;
  export default Plotly;
}
