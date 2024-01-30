<svelte:options accessors={true}/>
<script lang="ts">
    import CircularProgressbar from "./Components/CircularProgressbar.svelte"
    export let items: string[] = []
    export let progressStep: number = -1;
    let curProgress = 0;
    let curCount = 1;

    $: if (progressStep!==null) {
        incrementCurProgress()
    }
    //fake progress bar
    function incrementCurProgress(){
        curProgress =0;
        curCount = 100;

    }

    export let addStep = (newItem: string) => {
        items.push(newItem)
        progressStep+=1
    };

    export let complete = () => {
        curProgress=100
        clearTimeout(curInterval)
    };

    let curInterval = setInterval(()=>{
            curCount = curCount/2.5;
            curProgress += curCount;
        }, 500)
  </script>
  
  <div class="timeline-container">
    {#each items as item, index (index)}
      <div class="timeline-item {progressStep>=index?"activeInitial":""} {progressStep>index?"activeFinal":""}" >
        <CircularProgressbar class="timeline-circle" size="{28}" strokeWidth="{3}" progress="{progressStep==index?curProgress:(progressStep<=index?0:100)}"/>
        <div class="timeline-content">{item}</div>
      </div>
    {/each}
  </div>
  
  
  <style>
    .timeline-container {
      width: 100%;
      padding: 20px;
    }
  
    .timeline-item {
      display: flex;
      align-items: center;
      padding-bottom: 10px;
      position: relative;

    }

    .timeline-item:not(:last-child)::after{
        height: 50%;
        width: 2px;
        position: absolute;
        left: 13px;
        content: '';
        background-color: var(--color-light-grey);
        top: 50%;
        z-index: 0;
    }

    .timeline-item:not(:first-child)::before{
        height: 50%;
        width: 2px;
        position: absolute;
        left: 13px;
        content: '';
        background-color: var(--color-light-grey);
        top: 0%;
        z-index: 0;
    }

    .timeline-item.activeInitial::before{
        background-color: var(--color-photic-green);
    }

    .timeline-item.activeFinal::after{
        background-color: var(--color-photic-green);
    }
  
    .timeline-content {
      flex-grow: 1;
      padding-left: 20px;
      min-height: 40px;
      display: flex;
      align-items: center;
    }
  </style>
  