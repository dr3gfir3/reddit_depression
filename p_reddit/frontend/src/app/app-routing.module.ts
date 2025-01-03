import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { WordCountComponent } from './word-count/word-count.component';
import { EvaluateTextComponent } from './evaluate-text/evaluate-text.component';
import { DepressedComponent } from './depressed/depressed.component';
import { PostFrequencyComponent } from './post-frequency/post-frequency.component';

const routes: Routes = [
  { path: 'word-count', component: WordCountComponent },
  { path: 'evaluate-text', component: EvaluateTextComponent },
  { path: 'depressed-words', component: DepressedComponent },
  { path: 'post-frequency', component: PostFrequencyComponent },
  { path: '', redirectTo: '/word-count', pathMatch: 'full' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }