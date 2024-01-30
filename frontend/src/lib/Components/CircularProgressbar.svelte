<script>
    export let progress = 0; // Progress in percentage (0 to 100)
    export let size = 200; // Size of the SVG
    export let strokeWidth = 10; // Width of the stroke
    const radius = (size - strokeWidth) / 2;
    const circumference = 2 * Math.PI * radius;

    $: filledLength = (progress / 100) * circumference;
</script>

<svg
    {...$$restProps}
    width={size}
    height={size}
    viewBox={`0 0 ${size} ${size}`}
    xmlns="http://www.w3.org/2000/svg">
    <circle
        stroke="var(--color-light-grey)"
        fill="transparent"
        stroke-width={3}
        r={radius+1}
        cx={size / 2}
        cy={size / 2} />
    <circle
        fill="transparent"
        stroke="var(--color-photic-green)"
        stroke-width={size-3}
        stroke-dasharray={`${filledLength} ${circumference}`}
        transform={`rotate(-90) translate(-${size})`}
        r={radius}
        cx={size / 2}
        cy={size / 2}
        style="transition: stroke-dasharray 0.35s ease-out;"/>
</svg>
<style>
    svg {
        border-radius: 50%;
        display: block;
        z-index: 10;
        background-color: var(--color-white);
        flex-shrink: 0;
    }
</style>
