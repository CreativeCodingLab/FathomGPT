<script lang="ts">
    export let table: string;
    export let responseText: string;

    $: headers = table.length ? Object.keys(table[0]) : [];

</script>

<main>
    <h3>Fathom Said:</h3>
    <br/>
    <p>{responseText}</p>
    <br/>
    {#if typeof table === 'string'}
    <p>{table}</p>
    {:else}
    <table>
        <tr>
        {#each headers as header}
            <th>{header}</th>
        {/each}
        </tr>
        <!-- Data Rows -->
        {#each table as row}
            <tr>
                {#each headers as header}
                    <td>{row[header]}</td>
                {/each}
            </tr>
        {/each}
    </table>
    {/if}
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
    table {
        font-family: arial, sans-serif;
        border-collapse: collapse;
        width: 100%;
    }

    td, th {
    border: 1px solid #dddddd;
    text-align: left;
    padding: 8px;
    }

</style>