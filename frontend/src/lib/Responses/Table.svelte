<script lang="ts">
    export let table: string;
    export let responseText: string;

    $: headers = table.length ? Object.keys(table[0]) : [];

</script>

<main class="fathomChatContainer">
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