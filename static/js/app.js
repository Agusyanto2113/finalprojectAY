Array.from(document.getElementsByTagName('input')).forEach((e,i)=>{
    e.addEventListener('keyup',(e)=>{
        if (e.value.length > 0){
            document.getElementsByClassName('bi-caret-down-fill')[i].style.transform = "rotate(180deg)";
        }else{
            document.getElementsByClassName('bi-caret-down-fill')[i].style.transform = "rotate(0deg)";
        }
    })
})

let menu_btn = document.getElementsByClassName('bi-three-dots')[0];
let menu_bx = document.getElementById('menu_bx');

menu_btn.addEventListener('click', ()=>{
    menu_bx.classList.toggle('ul_active');
})



const startRecordingButton = document.getElementById('startButton');
const transcriptionDiv = document.getElementById('output');

let isRecording = false;
/*
        startRecordingButton.addEventListener('click', () => {
            if (!isRecording) {                
                isRecording = true;

                const recognition = new webkitSpeechRecognition();
                recognition.continuous = true;
                recognition.interimResults = true;
                recognition.lang = 'id-ID';

                recognition.onresult = function(event) {
                    const result = event.results[event.results.length - 1];
                    const transcript = result[0].transcript;
                    transcriptionDiv.innerHTML = transcript;

                    if (result.isFinal) {
                        // Send the audio file to the server for transcription
                        const formData = new FormData();
                        formData.append('audio', new Blob([transcript], { type: 'audio/wav' }));

                        fetch('/transcribe_audio/', {
                            method: 'POST',
                            body: formData,
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                console.log('Transcription saved:', data.text);
                            } else {
                                console.error('Transcription failed:', data.error);
                            }
                        });
                    }
                };

                recognition.start();
            
                // Stop recording after 3 seconds
                setTimeout(() => {
                    recognition.stop();
                    isRecording = false;
                    
                }, 3000); // 3 seconds

            } else {
                
                isRecording = false;
                recognition.stop();
            }
        });

*/


startRecordingButton.addEventListener('click', () => {
    if (!isRecording) {
        isRecording = true;

        const recognition = new webkitSpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'id-ID';

        recognition.onresult = function (event) {
            const result = event.results[event.results.length - 1];
            const transcript = result[0].transcript;
            transcriptionDiv.innerHTML = transcript;

            if (result.isFinal) {
                const formData = new FormData();
                formData.append('audio', new Blob([transcript], { type: 'audio/wav' }));

                fetch('/transcribe_audio/', {
                    method: 'POST',
                    body: formData,
                })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            console.log('Transcription saved:', data.text);
                        } else {
                            console.error('Transcription failed:', data.error);
                        }
                    });
            }
        };

        recognition.start();

        setTimeout(() => {
            recognition.stop();
            isRecording = false;
        }, 3000);
    } else {
        isRecording = false;
        recognition.stop();
    }
});


const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.getElementById('send-btn');
const chatbox = document.querySelector(".chatbox");
const chatbotToggler = document.querySelector('.chatbot-toggler');
const chatbotCloseBtn = document.querySelector('.close-btn');
const inputInitHeight = chatInput.scrollHeight;


let userMessage;


const createChatLi = (message,className)=>{
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat",className);
    let chatContent = className === "outgoing" ? `<p></p>`: `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    chatLi.innerHTML = chatContent;
    chatLi.querySelector("p").textContent = message;
    return chatLi;

}

function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = jQuery.trim(cookies[i]);
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

const generateResponse = (incomingChatLi) =>{

    const messageElement = incomingChatLi.querySelector("p");

    var csrfToken = $("meta[name='csrf-token']").attr("content");
    $.ajax({
        headers: {
            "X-CSRFToken": csrfToken
        },
        type: 'POST',
        url: '/chat_view/',  // URL of your Django API endpoint
        data: {'user_message': userMessage},
        success: function (data) {
            var response = data.response;
            // Handle the response (e.g., display it in the chat interface)
            //chatbox.appendChild(createChatLi(response,"incoming"));
            messageElement.textContent = response
            chatbox.scrollTo(0,chatbox.scrollHeight);
        },
        error: function (xhr, textStatus, error) {
            console.error('Error:', error);
            chatbox.scrollTo(0,chatbox.scrollHeight);
        }
    });
}


const handleChat = () =>{
    userMessage = chatInput.value.trim();
/*    console.log(userMessage);*/
    if(!userMessage) return;
    chatInput.value ="";
    /*chatInput.style.height = `${inputInitHeight}px`;*/


    chatbox.appendChild(createChatLi(userMessage,"outgoing"));
    chatbox.scrollTo(0,chatbox.scrollHeight);
    
    
    setTimeout(() => {
        const incomingChatLi = createChatLi(userMessage,"incoming");
        chatbox.appendChild(incomingChatLi);
        chatbox.scrollTo(0,chatbox.scrollHeight);
        generateResponse(incomingChatLi);
        
    }, 600);

}
/*
chatInput.addEventListener("input", () =>{
    chatInput.style.height = `${inputInitHeight}px`;
    chatInput.style.height = `${chatInput.scrollHeight}px`;
});

chatInput.addEventListener("keydown", (e) =>{
    if(e.key === "Enter" && !e.shiftKey && window.innerWidth > 800){
        e.preventDefault();
        handleChat();
    }
});
*/
sendChatBtn.addEventListener("click",handleChat);
chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));
chatbotCloseBtn.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));




