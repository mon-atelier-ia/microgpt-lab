import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  type ChartData,
  type ChartOptions,
  type Plugin,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

/** Chart.js plugin: adds canvas shadow glow behind each line dataset. */
const glowPlugin: Plugin<'line'> = {
  id: 'lineGlow',
  beforeDatasetDraw(chart, args) {
    const dataset = chart.data.datasets[args.index];
    const ctx = chart.ctx;
    ctx.save();
    ctx.shadowColor = (dataset.borderColor as string) ?? 'transparent';
    ctx.shadowBlur = args.index === 1 ? 12 : 6; // EMA (index 1) gets stronger glow
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
  },
  afterDatasetDraw(_chart, _args) {
    _chart.ctx.restore();
  },
};

type LossChartProps = {
  data: ChartData<'line'>;
  chartOptions: ChartOptions<'line'>;
};

export function LossChart({ data, chartOptions }: LossChartProps) {
  return <Line data={data} options={chartOptions} plugins={[glowPlugin]} />;
}
