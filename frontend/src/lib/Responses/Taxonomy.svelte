<script lang="ts">
	export let responseText: string;
	export let speciesArray: any[];

	console.log(speciesArray)

</script>

<main class="fathomChatContainer">
		<p>{responseText}</p>
		<br/>
		{#each speciesArray as species}
		<h3>Taxonomy for <b>{species.concept}</b></h3>
		{#if species != null}
		<div class="innerTaxonomyContainer">
			{#each species.taxonomy?.ancestors as ancestor, index}
			<div class="taxonomyItem" style="{'margin-left:'+(index*12)+"px"}"><b>{ancestor.name}</b> ({ancestor.rank})</div>
			{/each}
			<div class="taxonomyItem" style="{'margin-left:'+(species.taxonomy?.ancestors.length*12)+"px"}"><b>{species.concept}</b> ({species.rank})</div>
			{#each species.taxonomy?.descendants as descendant, index}
			<div class="taxonomyItem" style="{'margin-left:'+((species.taxonomy?.ancestors.length+index)*12)+"px"}"><b>{descendant.name}</b> ({descendant.rank})</div>
			{/each}
		</div>
		{/if}
		{/each}

</main>
