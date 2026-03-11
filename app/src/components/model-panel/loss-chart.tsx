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
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

type LossChartProps = {
  data: ChartData<'line'>;
  chartOptions: ChartOptions<'line'>;
};

export function LossChart({ data, chartOptions }: LossChartProps) {
  return <Line data={data} options={chartOptions} />;
}
