import Video from '$lib/Responses/Video.svelte';
export default function handleVideo(target: HTMLElement, jsonResponse: any): Video {

	return new Video({ target: target, props: { 
		videoUrl: jsonResponse.videoUrl,
		responseText: jsonResponse.responseText
		} });
}
