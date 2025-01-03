import { Component } from '@angular/core';
import { ApiService } from '../app/api/api.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'My Angular App';
  wordCount: number = 0;
  evaluation: string = '';
  depressedWords: string[] = [];
  postFrequency: number = 0;

  constructor(private apiService: ApiService) {}

  getWordCount() {
    this.apiService.wordCount().subscribe(data => {
      this.wordCount = data.word_count;
    });
  }

  evaluateText(text: string) {
    this.apiService.evaluateText(text).subscribe(data => {
      this.evaluation = data.evaluation;
    });
  }

  getDepressedWords() {
    this.apiService.depressedWords().subscribe(data => {
      this.depressedWords = data.depressed_words;
    });
  }

  getPostFrequency() {
    this.apiService.postFrequency().subscribe(data => {
      this.postFrequency = data.post_frequency;
    });
  }
}
