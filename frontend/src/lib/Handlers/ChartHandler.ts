import Chart from '$lib/Responses/Chart.svelte';

export default function handleChart(target: HTMLElement): Chart {

	return new Chart({
		target: target,
	});
}
