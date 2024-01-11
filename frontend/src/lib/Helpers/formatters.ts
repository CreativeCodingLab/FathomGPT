export const formattedDate = function(date:Date|null) {
    if (date==null) 
        return "";

    return new Date(date).toLocaleString('en-US', {
        day: 'numeric',
        month: 'short',
        year: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
    });
};