// filepath: /c:/Users/matti/Desktop/Progetto/reddit_depression/p_reddit/frontend/src/app/app.module.ts
import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';
import { AppRoutingModule } from './app-routing.module';

import { AppComponent } from './app.component';
import { ApiService } from './api/api.service';
import { WordCountComponent } from './word-count/word-count.component';
import { EvaluateTextComponent } from './evaluate-text/evaluate-text.component';
import { DepressedComponent } from './depressed/depressed.component';
import { PostFrequencyComponent } from './post-frequency/post-frequency.component';

@NgModule({
  declarations: [
    AppComponent,
    WordCountComponent,
    EvaluateTextComponent,
    DepressedComponent,
    PostFrequencyComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    AppRoutingModule
  ],
  providers: [ApiService],
  bootstrap: [AppComponent]
})
export class AppModule { }