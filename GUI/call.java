import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

private String callVQAModel(String base64Image, String question) {
    try {
        URL url = new URL("http://127.0.0.1:8000/vqa/");  
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
        conn.setDoOutput(true);

        String data = "image_base64=" + base64Image + "&question=" + question;
        OutputStream os = conn.getOutputStream();
        os.write(data.getBytes(StandardCharsets.UTF_8));
        os.flush();
        os.close();

        Scanner scanner = new Scanner(conn.getInputStream(), StandardCharsets.UTF_8);
        String response = scanner.useDelimiter("\\A").next();
        scanner.close();

        return response;  
    } catch (Exception e) {
        e.printStackTrace();
        return "Error connecting to VQA model.";
    }
}
