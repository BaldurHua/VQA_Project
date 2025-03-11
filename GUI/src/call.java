import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;

private String callVQAModel(String base64Image, String question) {
    try {
        HttpClient client = HttpClient.newHttpClient();
        String data = "image_base64=" + base64Image + "&question=" + question;

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("http://127.0.0.1:8000/vqa/"))  
                .header("Content-Type", "application/x-www-form-urlencoded")
                .POST(HttpRequest.BodyPublishers.ofString(data, StandardCharsets.UTF_8))
                .build();

        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        return response.body(); 

    } catch (Exception e) {
        e.printStackTrace();
        return "Error connecting to VQA model.";
    }
}
