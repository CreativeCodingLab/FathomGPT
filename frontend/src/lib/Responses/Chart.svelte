<script lang="ts">
    import Chart from 'chart.js/auto';
	import { onMount } from 'svelte';
// JavaScript: Define the data points
const data: { x: number; y: number }[] = [
  { x: 0, y: Math.random() * 10 },
  { x: 1, y: Math.random() * 10 },
  { x: 2, y: Math.random() * 10 },
  { x: 3, y: Math.random() * 10 },
  { x: 4, y: Math.random() * 10 },
  { x: 5, y: Math.random() * 10 },
  { x: 6, y: Math.random() * 10 },
  { x: 7, y: Math.random() * 10 }
];

let myChart: HTMLCanvasElement;

// TypeScript: Create a function to generate the line chart
function createLineChart(data: { x: number; y: number }[]) { 
  console.log(myChart);
  if (myChart) {
    const chart = new Chart(myChart, {
      type: 'line',
      data: {
        datasets: [{
          label: 'Data',
          data: data,
          borderColor: 'blue',
          fill: false
        }]
      },
      options: {
        scales: {
          x: {
            type: 'linear',
            position: 'bottom'
          },
          y: {
            type: 'linear',
            position: 'left'
          }
        }
      }
    });
  }
}

// Call the function to create the chart
onMount(() => {
  createLineChart(data);
  window.scrollTo(0, document.body.scrollHeight);
});
</script>

<main>
  <h3>Fathom said:</h3>
  <canvas bind:this ={myChart} width="400" height="200"></canvas>
</main>

<style>
    main {
      width: 100%;
      border-radius: 0.5rem;
      padding: 0.5rem;
      background-color: var(--background-light);
      min-height: 5rem;
      color: white;
      display: grid;
      place-items: center start;
    }
</style>