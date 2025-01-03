export class ApiService {
  private baseUrl: string = 'http://localhost:8000'; // Adjust the base URL as needed

  constructor(private http: HttpClient) {}

  wordCount() {
    return this.http.get<{ word_count: number }>(`${this.baseUrl}/word_count/`);
  }

  evaluateText(text: string) {
    return this.http.get<{ evaluation: any }>(`${this.baseUrl}/evaluate_text/?text=${encodeURIComponent(text)}`);
  }

  depressedWords() {
    return this.http.get<{ depressed_words: any }>(`${this.baseUrl}/depressed_words/`);
  }

  postFrequency() {
    return this.http.get<{ post_frequency: any }>(`${this.baseUrl}/post_frequency/`);
  }
}