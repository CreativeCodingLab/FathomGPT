<script lang="ts">
	import type { speciesData } from "$lib/types/responseType";

	export let responseText: string;
	export let speciesArray: any[];

	let indent = 20;
	let descPadding = 0;
	//let lastRank = speciesArray[0].taxonomy.descendants[0].rank;
</script>

<main>
		<h3>Fathom said:</h3>
		<p>{responseText}</p>
		<br />
		<!-- {#each speciesArray as s}
			<h3>{s.concept}</h3>
			<ul>
				{#each s.taxonomy.ancestors as a, i}
					<li style="padding-left: {s.taxonomy.ancestors.length*indent - i*indent}px;">{a.name} : {a.rank}</li>
				{/each}
				<h3>{s.concept}</h3>
				{#each s.taxonomy.descendants as d, i}
					<li style="padding-left: {d.rank !== lastRank ? descPadding +=indent : descPadding}px;">{d.name} : {d.rank}</li>
				{/each}
			</ul>
		{/each} -->
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
